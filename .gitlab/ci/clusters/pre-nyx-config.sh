# Standalone NCCL EP CI configuration for Pre-Nyx (B200).

PRENYX_NCCL_SOCKET_IFNAME="ens6f1np1"

function configure_test_env() {
    export UCX_NET_DEVICES=$PRENYX_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp
    export NCCL_IB_HCA="=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"
    export OMPI_MCA_rmaps_oversubscribe=1
    export OMPI_MCA_rmaps_binding_policy=none
    export OMPI_MCA_hwloc_base_binding_policy=none
    export OMPI_MCA_btl="tcp,self"
    export OMPI_MCA_btl_tcp_if_include=$PRENYX_NCCL_SOCKET_IFNAME
}
