# Config parameters for ipp6 cluster
# No host-specific code executes in this module to allow it to be served over http in the future
#
# Targeting build to run in ipp6 environment (A100/A40/L40S/H100)
IPP6_GPU_ARCHS="80,86,89,90"

IPP6_OS_VERSION="20.04"
IPP6_CUDA_VERSION="12.8.0"
IPP6_BUILD_TOOLS_VERSION="2.1.1"
IPP6_BUILD_IMAGE_VERSION="${IPP6_BUILD_TOOLS_VERSION}-c${IPP6_CUDA_VERSION}-u${IPP6_OS_VERSION}"

IPP6_TOOLKIT_DIR="/storage/toolkits"
IPP6_CUDA_HOME="$IPP6_TOOLKIT_DIR/cuda-$IPP6_CUDA_VERSION"
IPP6_OPENMPI_HOME="/cm/shared/apps/openmpi4/gcc/4.1.8"

IPP6_DOCKER_IMAGE_DIR="$IPP6_TOOLKIT_DIR/docker_sqsh"

IPP6_PLANNED_RESERVED="Skip"

# Set interface name based on partition
# Use a100 interface as default
IPP6_NCCL_SOCKET_IFNAME="enp134s0np0"
if [[ "$SLURM_PARTITION" =~ ^a100 ]]; then
    IPP6_NCCL_SOCKET_IFNAME="enp134s0np0"
elif [[ "$SLURM_PARTITION" =~ ^a40 ]]; then
    IPP6_NCCL_SOCKET_IFNAME="enp65s0np0"
elif [[ "$SLURM_PARTITION" =~ ^l40s ]]; then
    IPP6_NCCL_SOCKET_IFNAME="ens255f0np0"
elif [[ "$SLURM_PARTITION" =~ ^h100 ]]; then
    IPP6_NCCL_SOCKET_IFNAME="ens255np0"
fi

# Target configs
function get_gpu_archs() {
    echo "$IPP6_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$IPP6_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$IPP6_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$IPP6_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_num_build_procs() {
    # ignore on ipp6
    hostname="$1"

    echo "nproc"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    echo "srun \
        -p cpuonly \
        -J nccl:build \
        -t ${BUILD_SLURM_TIME:-00:30:00} \
        -n 1 \
        --exclusive \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_docs_build_command() {
    current_dir="$1"
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    echo "srun \
        -p cpuonly \
        -J nccl:docs-build \
        -t 00:10:00 \
        -n 1 \
        --exclusive \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        make -C /nccl pkg.doc.build"
}

function get_cuda_home() {
    echo "$IPP6_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$IPP6_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$IPP6_CUDA_HOME/lib64"
}

function get_extra_path() {
    echo "$IPP6_CUDA_HOME/bin"
}

function get_planned_reserved() {
    echo "$IPP6_PLANNED_RESERVED"
}

function get_nccl_socket_ifname() {
    echo "$IPP6_NCCL_SOCKET_IFNAME"
}

function get_mpi_params() {
    echo "--oversubscribe --bind-to none --mca btl tcp,self --mca btl_tcp_if_include $IPP6_NCCL_SOCKET_IFNAME"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$IPP6_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp
    export OMPI_MCA_btl="^openib"
    export OMPI_MCA_rmaps_oversubscribe=1
    export OMPI_MCA_rmaps_binding_policy=none
    export OMPI_MCA_hwloc_base_binding_policy=none
    export OMPI_MCA_btl="tcp,self"
    export OMPI_MCA_btl_tcp_if_include=$IPP6_NCCL_SOCKET_IFNAME

    # Note: These are the E/W IB interfaces on Pre-Nyx
    # export NCCL_IB_HCA="=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"
}
