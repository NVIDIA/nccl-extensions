# Standalone NCCL EP CI configuration for Pre-Tyche (GB200).

PRETYCHE_GPU_ARCHS="100"
PRETYCHE_CUDA_VERSION="13.0.2"
PRETYCHE_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-13.0"
PRETYCHE_OPENMPI_VERSION="5.0.6"
PRETYCHE_OPENMPI_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/openmpi-${PRETYCHE_OPENMPI_VERSION}"
PRETYCHE_OS_VERSION="24.04"
PRETYCHE_BUILD_TOOLS_VERSION="2.0.0"
PRETYCHE_BUILD_IMAGE_VERSION="${PRETYCHE_BUILD_TOOLS_VERSION}-c${PRETYCHE_CUDA_VERSION}-u${PRETYCHE_OS_VERSION}"
PRETYCHE_DOCKER_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"
PRETYCHE_SLURM_ACCOUNT="coreai_libraries_nccl"
PRETYCHE_NCCL_SOCKET_IFNAME="enP6p3s0f1np1"

function get_gpu_archs() {
    echo "$PRETYCHE_GPU_ARCHS"
}

function get_slurm_account() {
    echo "$PRETYCHE_SLURM_ACCOUNT"
}

function get_build_image_version() {
    echo "$PRETYCHE_BUILD_IMAGE_VERSION"
}

function get_build_tools_image() {
    build_image_version="$1"
    echo "$PRETYCHE_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$PRETYCHE_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp
    export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_4,mlx5_5"
    export OMPI_MCA_rmaps_oversubscribe=1
    export OMPI_MCA_rmaps_binding_policy=none
    export OMPI_MCA_hwloc_base_binding_policy=none
    export OMPI_MCA_btl="tcp,self"
    export OMPI_MCA_btl_tcp_if_include=$PRETYCHE_NCCL_SOCKET_IFNAME
}
