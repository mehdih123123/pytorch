from typing import cast, List

from ... import config
from ...codecache import code_hash, get_path
from ...scheduler import FusedSchedulerNode, Scheduler, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import IndentedBuffer
from ..triton import TritonScheduling
from .cuda_kernel import CUDATemplateBuffer


class CUDASchedulerNode(SchedulerNode):
    def __init__(self, scheduler: Scheduler, node: CUDATemplateBuffer, group_fn):
        assert isinstance(node, CUDATemplateBuffer)
        super().__init__(scheduler, node, group_fn)
        self.group_fn = group_fn  # keeping this to enable cloning during fuse_epilogue

    def can_fuse_epilogue(self, other_node: SchedulerNode) -> bool:
        cnode: CUDATemplateBuffer = cast(CUDATemplateBuffer, self.node)
        if other_node.get_device() != self.get_device():
            return False
        return cnode.can_fuse_epilogue(other_node.node)

    def fuse_epilogue(self, other_node: SchedulerNode) -> FusedSchedulerNode:
        assert self.can_fuse_epilogue(other_node)
        epilogue_nodes = [other_node]

        cnode: CUDATemplateBuffer = cast(CUDATemplateBuffer, self.node)
        fused_template_buffer = cnode.create_fused_buffer(
            other_node.node, self.scheduler
        )
        self.node = fused_template_buffer
        fused_node = FusedCUDASchedulerNode(self, self.scheduler, epilogue_nodes)
        return fused_node


class FusedCUDASchedulerNode(FusedSchedulerNode):
    def __init__(
        self,
        cuda_scheduler_node: CUDASchedulerNode,
        scheduler: Scheduler,
        epilogue_scheduler_nodes: List[SchedulerNode],
    ):
        assert isinstance(cuda_scheduler_node, CUDASchedulerNode)
        assert (
            cuda_scheduler_node not in epilogue_scheduler_nodes
        ), "template node should not be in snodes"
        assert cuda_scheduler_node.is_template()
        super().__init__(scheduler, [cuda_scheduler_node] + epilogue_scheduler_nodes)
        self.template_node = cuda_scheduler_node

    def get_cuda_scheduler_node(self) -> CUDASchedulerNode:
        return self.template_node

    def get_epilogue_nodes(self) -> List[SchedulerNode]:
        return self.snodes[1:]

    def can_fuse_epilogue(self, node: SchedulerNode) -> bool:
        return self.get_cuda_scheduler_node().can_fuse_epilogue(node)

    def fuse_epilogue(self, other_node: SchedulerNode) -> FusedSchedulerNode:
        assert self.can_fuse_epilogue(other_node)
        # Non-mutating, the fused nodes are new instances
        tnode: CUDATemplateBuffer = self.get_cuda_scheduler_node().node
        fused_template_buffer = tnode.create_fused_buffer(
            other_node.node, self.scheduler
        )
        template_node = CUDASchedulerNode(
            self.scheduler,
            fused_template_buffer,
            self.get_cuda_scheduler_node().group_fn,
        )
        fused_node = FusedCUDASchedulerNode(
            template_node, self.scheduler, self.get_epilogue_nodes() + [other_node]
        )
        return fused_node


class CUDAScheduling(TritonScheduling):
    """
    Final codegen for CUDAKernels.
    """

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cuda", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.cuda(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''', 'so')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(self, template_node, epilogue_nodes):
        """
        Codegen a CUDA template
        """
        assert isinstance(
            template_node, CUDASchedulerNode
        ), "Template node passed to CUDAScheduler.codegen_template must be a CUDASchedulerNode."
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CUDATemplateBuffer = template_node.node
        assert {n.get_name() for n in epilogue_nodes} == {
            n.get_name() for n in ctb.epilogue_nodes
        }, "Epilogue node set not identical. This should never happen."

        kernel, render = ctb.make_kernel_render(template_node.node)
        epilogue_init_list = []
        epilogue_param_list = []
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()
