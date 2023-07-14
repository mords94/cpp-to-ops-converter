#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
WORKPATH=$scriptDir/../

KERNELS=(
    ### KERNELS
    # "ext_adjust_u_v_"
    # "ext_advave_"
    # "ext_advq_"
    # "ext_apply_filter_"
    # "ext_baropg_"
    # "ext_bcond_1_"
    # "ext_bcond_2_"
    # "ext_bcond_5_"
    # "ext_comp_vamax_"
    # "ext_dens_"
    # "ext_elf_update_"
    # "ext_etf_"
    # "ext_flux_update_"
    # "ext_init_horizontal_velocities_"
    # "ext_init_internal_"
    # "ext_uaf_"
    # "ext_vaf_"
    # "ext_vertvl_"

    # "ext_advv_"
    # "ext_advu_"
    # "ext_updeta_t_s_"
    # "ext_update_u_v_"
    # "ext_vert_avgs_"
    # "ext_add_ad_2d_"
    # "ext_time_internal_forward_"
    # "ext_advct_"
    # "ext_update_turbulane_"
    # "ext_bcond_3_"

    # "ext_final_internal_update_"
    # "ext_smol_adif_"
    # "ext_advt1_"
    # "ext_advt2_"
    # "ext_aam_"

    "ext_proft_"
    "ext_profu_"
    "ext_profv_"
    "ext_profq_"

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
    $WORKPATH/script/compile.sh $kernel
done
