#!/bin/bash

WORKPATH=/Users/mac/Projects/mlir/cpp-to-ops-converter
KERNELS=(
    ### KERNELS
    "ext_adjust_u_v_"
    "ext_advave_"
    "ext_advq_"
    "ext_apply_filter_"
    "ext_baropg_"
    "ext_bcond_1_"
    "ext_bcond_2_"
    "ext_bcond_5_"
    "ext_comp_vamax_"
    "ext_dens_"
    "ext_elf_update_"
    "ext_etf_"
    "ext_flux_update_"
    "ext_init_horizontal_velocities_"
    "ext_init_internal_"

    "ext_uaf_"
    "ext_vaf_"
    "ext_vertvl_"

    ### IO
    # "ext_load_state_for_restart_" # load
    # "ext_save_2d_array_" # save
    # "ext_save_3d_array_" # save
    # "ext_save_state_for_restart_" # save

    ### INIT
    # "ext_areas_masks_" # init
    # "ext_depth_" # init
    # "ext_init_cond_" # init
    # "ext_init_cond2_" # init
    # "ext_init_cond3_" # init
    # "ext_seamount_" # init
    # "ext_slpmax_" # init
)

for kernel in "${KERNELS[@]}"; do
    $WORKPATH/script/compile.sh $kernel --opconv-debug
done
