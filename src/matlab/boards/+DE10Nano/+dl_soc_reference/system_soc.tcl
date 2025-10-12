# Add clock constraint
set_location_assignment PIN_AN18 -to clk_clk
set_location_assignment PIN_AG5 -to clk_133m_clk
set_location_assignment PIN_AG6 -to "clk_133m_clk(n)"

# Add DDR constraint
set_location_assignment PIN_AN3 -to ddr4_emif_mem_mem_a[0]
set_location_assignment PIN_AJ5 -to ddr4_emif_mem_mem_a[10]
set_location_assignment PIN_AH6 -to ddr4_emif_mem_mem_a[11]
set_location_assignment PIN_AG7 -to ddr4_emif_mem_mem_a[12]
set_location_assignment PIN_AJ3 -to ddr4_emif_mem_mem_a[13]
set_location_assignment PIN_AH3 -to ddr4_emif_mem_mem_a[14]
set_location_assignment PIN_AF7 -to ddr4_emif_mem_mem_a[15]
set_location_assignment PIN_AE7 -to ddr4_emif_mem_mem_a[16]
set_location_assignment PIN_AM4 -to ddr4_emif_mem_mem_a[1]
set_location_assignment PIN_AL3 -to ddr4_emif_mem_mem_a[2]
set_location_assignment PIN_AL4 -to ddr4_emif_mem_mem_a[3]
set_location_assignment PIN_AL5 -to ddr4_emif_mem_mem_a[4]
set_location_assignment PIN_AK5 -to ddr4_emif_mem_mem_a[5]
set_location_assignment PIN_AK6 -to ddr4_emif_mem_mem_a[6]
set_location_assignment PIN_AJ6 -to ddr4_emif_mem_mem_a[7]
set_location_assignment PIN_AK3 -to ddr4_emif_mem_mem_a[8]
set_location_assignment PIN_AJ4 -to ddr4_emif_mem_mem_a[9]
set_location_assignment PIN_AL2 -to ddr4_emif_mem_mem_act_n[0]
set_location_assignment PIN_AF9 -to ddr4_emif_mem_mem_alert_n[0]
set_location_assignment PIN_AF5 -to ddr4_emif_mem_mem_ba[0]
set_location_assignment PIN_AH4 -to ddr4_emif_mem_mem_ba[1]
set_location_assignment PIN_AG4 -to ddr4_emif_mem_mem_bg[0]
set_location_assignment PIN_AK1 -to ddr4_emif_mem_mem_ck[0]
set_location_assignment PIN_AK2 -to ddr4_emif_mem_mem_ck_n[0]
set_location_assignment PIN_AM1 -to ddr4_emif_mem_mem_cke[0]
set_location_assignment PIN_AM2 -to ddr4_emif_mem_mem_cs_n[0]
set_location_assignment PIN_AH8 -to ddr4_emif_mem_mem_dbi_n[0]
set_location_assignment PIN_AM6 -to ddr4_emif_mem_mem_dbi_n[1]
set_location_assignment PIN_AM5 -to ddr4_emif_mem_mem_dbi_n[2]
set_location_assignment PIN_AT4 -to ddr4_emif_mem_mem_dbi_n[3]
set_location_assignment PIN_AA10 -to ddr4_emif_mem_mem_dbi_n[4]
set_location_assignment PIN_AB5 -to ddr4_emif_mem_mem_dbi_n[5]
set_location_assignment PIN_AB2 -to ddr4_emif_mem_mem_dbi_n[6]
set_location_assignment PIN_AC1 -to ddr4_emif_mem_mem_dbi_n[7]
set_location_assignment PIN_AE11 -to ddr4_emif_mem_mem_dbi_n[8]
set_location_assignment PIN_AG12 -to ddr4_emif_mem_mem_dq[0]
set_location_assignment PIN_AK10 -to ddr4_emif_mem_mem_dq[10]
set_location_assignment PIN_AL9 -to ddr4_emif_mem_mem_dq[11]
set_location_assignment PIN_AN6 -to ddr4_emif_mem_mem_dq[12]
set_location_assignment PIN_AK7 -to ddr4_emif_mem_mem_dq[13]
set_location_assignment PIN_AM9 -to ddr4_emif_mem_mem_dq[14]
set_location_assignment PIN_AL7 -to ddr4_emif_mem_mem_dq[15]
set_location_assignment PIN_AR3 -to ddr4_emif_mem_mem_dq[16]
set_location_assignment PIN_AU2 -to ddr4_emif_mem_mem_dq[17]
set_location_assignment PIN_AP4 -to ddr4_emif_mem_mem_dq[18]
set_location_assignment PIN_AP3 -to ddr4_emif_mem_mem_dq[19]
set_location_assignment PIN_AJ9 -to ddr4_emif_mem_mem_dq[1]
set_location_assignment PIN_AN4 -to ddr4_emif_mem_mem_dq[20]
set_location_assignment PIN_AU1 -to ddr4_emif_mem_mem_dq[21]
set_location_assignment PIN_AP5 -to ddr4_emif_mem_mem_dq[22]
set_location_assignment PIN_AT3 -to ddr4_emif_mem_mem_dq[23]
set_location_assignment PIN_AU4 -to ddr4_emif_mem_mem_dq[24]
set_location_assignment PIN_AW5 -to ddr4_emif_mem_mem_dq[25]
set_location_assignment PIN_AU5 -to ddr4_emif_mem_mem_dq[26]
set_location_assignment PIN_AV4 -to ddr4_emif_mem_mem_dq[27]
set_location_assignment PIN_AW4 -to ddr4_emif_mem_mem_dq[28]
set_location_assignment PIN_AR6 -to ddr4_emif_mem_mem_dq[29]
set_location_assignment PIN_AH9 -to ddr4_emif_mem_mem_dq[2]
set_location_assignment PIN_AR7 -to ddr4_emif_mem_mem_dq[30]
set_location_assignment PIN_AT5 -to ddr4_emif_mem_mem_dq[31]
set_location_assignment PIN_Y8 -to ddr4_emif_mem_mem_dq[32]
set_location_assignment PIN_AB11 -to ddr4_emif_mem_mem_dq[33]
set_location_assignment PIN_AB10 -to ddr4_emif_mem_mem_dq[34]
set_location_assignment PIN_AB9 -to ddr4_emif_mem_mem_dq[35]
set_location_assignment PIN_W8 -to ddr4_emif_mem_mem_dq[36]
set_location_assignment PIN_Y10 -to ddr4_emif_mem_mem_dq[37]
set_location_assignment PIN_AA9 -to ddr4_emif_mem_mem_dq[38]
set_location_assignment PIN_AB7 -to ddr4_emif_mem_mem_dq[39]
set_location_assignment PIN_AF12 -to ddr4_emif_mem_mem_dq[3]
set_location_assignment PIN_Y6 -to ddr4_emif_mem_mem_dq[40]
set_location_assignment PIN_Y7 -to ddr4_emif_mem_mem_dq[41]
set_location_assignment PIN_AA5 -to ddr4_emif_mem_mem_dq[42]
set_location_assignment PIN_Y5 -to ddr4_emif_mem_mem_dq[43]
set_location_assignment PIN_AD4 -to ddr4_emif_mem_mem_dq[44]
set_location_assignment PIN_AC6 -to ddr4_emif_mem_mem_dq[45]
set_location_assignment PIN_AD5 -to ddr4_emif_mem_mem_dq[46]
set_location_assignment PIN_AB6 -to ddr4_emif_mem_mem_dq[47]
set_location_assignment PIN_AB4 -to ddr4_emif_mem_mem_dq[48]
set_location_assignment PIN_W1 -to ddr4_emif_mem_mem_dq[49]
set_location_assignment PIN_AH11 -to ddr4_emif_mem_mem_dq[4]
set_location_assignment PIN_Y1 -to ddr4_emif_mem_mem_dq[50]
set_location_assignment PIN_AA4 -to ddr4_emif_mem_mem_dq[51]
set_location_assignment PIN_Y3 -to ddr4_emif_mem_mem_dq[52]
set_location_assignment PIN_AB1 -to ddr4_emif_mem_mem_dq[53]
set_location_assignment PIN_Y2 -to ddr4_emif_mem_mem_dq[54]
set_location_assignment PIN_AC4 -to ddr4_emif_mem_mem_dq[55]
set_location_assignment PIN_AE3 -to ddr4_emif_mem_mem_dq[56]
set_location_assignment PIN_AE2 -to ddr4_emif_mem_mem_dq[57]
set_location_assignment PIN_AE1 -to ddr4_emif_mem_mem_dq[58]
set_location_assignment PIN_AF3 -to ddr4_emif_mem_mem_dq[59]
set_location_assignment PIN_AG11 -to ddr4_emif_mem_mem_dq[5]
set_location_assignment PIN_AG2 -to ddr4_emif_mem_mem_dq[60]
set_location_assignment PIN_AF2 -to ddr4_emif_mem_mem_dq[61]
set_location_assignment PIN_AD3 -to ddr4_emif_mem_mem_dq[62]
set_location_assignment PIN_AD1 -to ddr4_emif_mem_mem_dq[63]
set_location_assignment PIN_AD9 -to ddr4_emif_mem_mem_dq[64]
set_location_assignment PIN_AE10 -to ddr4_emif_mem_mem_dq[65]
set_location_assignment PIN_AC8 -to ddr4_emif_mem_mem_dq[66]
set_location_assignment PIN_AC9 -to ddr4_emif_mem_mem_dq[67]
set_location_assignment PIN_AD8 -to ddr4_emif_mem_mem_dq[68]
set_location_assignment PIN_AC11 -to ddr4_emif_mem_mem_dq[69]
set_location_assignment PIN_AJ8 -to ddr4_emif_mem_mem_dq[6]
set_location_assignment PIN_AD10 -to ddr4_emif_mem_mem_dq[70]
set_location_assignment PIN_AF10 -to ddr4_emif_mem_mem_dq[71]
set_location_assignment PIN_AJ11 -to ddr4_emif_mem_mem_dq[7]
set_location_assignment PIN_AK8 -to ddr4_emif_mem_mem_dq[8]
set_location_assignment PIN_AL8 -to ddr4_emif_mem_mem_dq[9]
set_location_assignment PIN_AG9 -to ddr4_emif_mem_mem_dqs[0]
set_location_assignment PIN_AN7 -to ddr4_emif_mem_mem_dqs[1]
set_location_assignment PIN_AR5 -to ddr4_emif_mem_mem_dqs[2]
set_location_assignment PIN_AW6 -to ddr4_emif_mem_mem_dqs[3]
set_location_assignment PIN_AA7 -to ddr4_emif_mem_mem_dqs[4]
set_location_assignment PIN_AE5 -to ddr4_emif_mem_mem_dqs[5]
set_location_assignment PIN_AA2 -to ddr4_emif_mem_mem_dqs[6]
set_location_assignment PIN_AH1 -to ddr4_emif_mem_mem_dqs[7]
set_location_assignment PIN_AF8 -to ddr4_emif_mem_mem_dqs[8]
set_location_assignment PIN_AG10 -to ddr4_emif_mem_mem_dqs_n[0]
set_location_assignment PIN_AM7 -to ddr4_emif_mem_mem_dqs_n[1]
set_location_assignment PIN_AP6 -to ddr4_emif_mem_mem_dqs_n[2]
set_location_assignment PIN_AV6 -to ddr4_emif_mem_mem_dqs_n[3]
set_location_assignment PIN_AA8 -to ddr4_emif_mem_mem_dqs_n[4]
set_location_assignment PIN_AE6 -to ddr4_emif_mem_mem_dqs_n[5]
set_location_assignment PIN_AA3 -to ddr4_emif_mem_mem_dqs_n[6]
set_location_assignment PIN_AG1 -to ddr4_emif_mem_mem_dqs_n[7]
set_location_assignment PIN_AE8 -to ddr4_emif_mem_mem_dqs_n[8]
set_location_assignment PIN_AR1 -to ddr4_emif_mem_mem_odt[0]
set_location_assignment PIN_AH2 -to ddr4_emif_mem_mem_par[0]
set_location_assignment PIN_AN2 -to ddr4_emif_mem_mem_reset_n[0]

set_location_assignment PIN_AH7 -to ddr4_emif_oct_oct_rzqin

set_location_assignment PIN_AV21 -to reset_133m_reset_n
	
set_location_assignment PIN_AR23 -to ddr4_emif_status_local_cal_fail
set_location_assignment PIN_AR22 -to ddr4_emif_status_local_cal_success

set_location_assignment PIN_AM21 -to ddr4_emif_ctrl_ecc_user_interrupt_0_ctrl_ecc_user_interrupt



# ....................................................................

# Add HPS constraint
set_location_assignment PIN_F23 -to hps_mem_mem_a[16]
set_location_assignment PIN_F24 -to hps_mem_mem_a[15]
set_location_assignment PIN_G25 -to hps_mem_mem_a[14]
set_location_assignment PIN_G26 -to hps_mem_mem_a[13]
set_location_assignment PIN_F26 -to hps_mem_mem_a[12]
set_location_assignment PIN_D24 -to hps_mem_mem_a[11]
set_location_assignment PIN_C24 -to hps_mem_mem_a[10]
set_location_assignment PIN_E23 -to hps_mem_mem_a[9]
set_location_assignment PIN_D23 -to hps_mem_mem_a[8]
set_location_assignment PIN_C23 -to hps_mem_mem_a[7]
set_location_assignment PIN_B24 -to hps_mem_mem_a[5]
set_location_assignment PIN_B22 -to hps_mem_mem_a[6]
set_location_assignment PIN_C25 -to hps_mem_mem_a[4]
set_location_assignment PIN_C21 -to hps_mem_mem_a[3]
set_location_assignment PIN_C22 -to hps_mem_mem_a[2]
set_location_assignment PIN_C26 -to hps_mem_mem_a[1]
set_location_assignment PIN_B26 -to hps_mem_mem_a[0]
set_location_assignment PIN_AJ26 -to hps_mem_mem_dq[31]
set_location_assignment PIN_AJ23 -to hps_mem_mem_dq[30]
set_location_assignment PIN_AJ24 -to hps_mem_mem_dq[29]
set_location_assignment PIN_AF25 -to hps_mem_mem_dq[28]
set_location_assignment PIN_AF24 -to hps_mem_mem_dq[27]
set_location_assignment PIN_AG25 -to hps_mem_mem_dq[26]
set_location_assignment PIN_AH23 -to hps_mem_mem_dq[25]
set_location_assignment PIN_AH24 -to hps_mem_mem_dq[24]
set_location_assignment PIN_AV27 -to hps_mem_mem_dq[23]
set_location_assignment PIN_AV28 -to hps_mem_mem_dq[22]
set_location_assignment PIN_AW24 -to hps_mem_mem_dq[21]
set_location_assignment PIN_AV24 -to hps_mem_mem_dq[20]
set_location_assignment PIN_AW28 -to hps_mem_mem_dq[19]
set_location_assignment PIN_AV23 -to hps_mem_mem_dq[18]
set_location_assignment PIN_AU27 -to hps_mem_mem_dq[17]
set_location_assignment PIN_AU28 -to hps_mem_mem_dq[16]
set_location_assignment PIN_AU26 -to hps_mem_mem_dq[15]
set_location_assignment PIN_AU24 -to hps_mem_mem_dq[14]
set_location_assignment PIN_AP25 -to hps_mem_mem_dq[13]
set_location_assignment PIN_AT23 -to hps_mem_mem_dq[12]
set_location_assignment PIN_AR25 -to hps_mem_mem_dq[11]
set_location_assignment PIN_AR26 -to hps_mem_mem_dq[10]
set_location_assignment PIN_AT26 -to hps_mem_mem_dq[9]
set_location_assignment PIN_AP23 -to hps_mem_mem_dq[8]
set_location_assignment PIN_AK23 -to hps_mem_mem_dq[7]
set_location_assignment PIN_AL23 -to hps_mem_mem_dq[5]
set_location_assignment PIN_AL26 -to hps_mem_mem_dq[6]
set_location_assignment PIN_AK26 -to hps_mem_mem_dq[4]
set_location_assignment PIN_AM24 -to hps_mem_mem_dq[3]
set_location_assignment PIN_AN23 -to hps_mem_mem_dq[2]
set_location_assignment PIN_AN24 -to hps_mem_mem_dq[1]
set_location_assignment PIN_AP26 -to hps_mem_mem_dq[0]
set_location_assignment PIN_AJ25 -to hps_mem_mem_dqs_n[3]
set_location_assignment PIN_AK25 -to hps_mem_mem_dqs[3]
set_location_assignment PIN_AW25 -to hps_mem_mem_dqs_n[2]
set_location_assignment PIN_AW26 -to hps_mem_mem_dqs[2]
set_location_assignment PIN_AT24 -to hps_mem_mem_dqs_n[1]
set_location_assignment PIN_AT25 -to hps_mem_mem_dqs[1]
set_location_assignment PIN_AL25 -to hps_mem_mem_dqs_n[0]
set_location_assignment PIN_AM25 -to hps_mem_mem_dqs[0]

set_location_assignment PIN_F25 -to pll_ref_clk_clk
set_location_assignment PIN_G24 -to "pll_ref_clk_clk(n)"
set_instance_assignment -name IO_STANDARD LVDS -to pll_ref_clk_clk
set_instance_assignment -name IO_STANDARD LVDS -to "pll_ref_clk_clk(n)"

set_location_assignment PIN_E26 -to hps_oct_oct_rzqin

set_location_assignment PIN_H24 -to hps_mem_mem_ba[1]
set_location_assignment PIN_E25 -to hps_mem_mem_ba[0]
set_location_assignment PIN_AH25 -to hps_mem_mem_dbi_n[3]
set_location_assignment PIN_AV26 -to hps_mem_mem_dbi_n[2]
set_location_assignment PIN_AU25 -to hps_mem_mem_dbi_n[1]
set_location_assignment PIN_AN26 -to hps_mem_mem_dbi_n[0]

set_location_assignment PIN_E22 -to hps_io_hps_io_phery_usb0_DATA7
set_location_assignment PIN_E21 -to hps_io_hps_io_phery_usb0_DATA6
set_location_assignment PIN_D20 -to hps_io_hps_io_phery_usb0_DATA5
set_location_assignment PIN_D21 -to hps_io_hps_io_phery_usb0_DATA4
set_location_assignment PIN_C18 -to hps_io_hps_io_phery_usb0_DATA3
set_location_assignment PIN_C17 -to hps_io_hps_io_phery_usb0_DATA2
set_location_assignment PIN_E17 -to hps_io_hps_io_phery_usb0_DATA1
set_location_assignment PIN_D19 -to hps_io_hps_io_phery_usb0_DATA0

set_location_assignment PIN_J24 -to hps_mem_mem_bg
set_location_assignment PIN_B21 -to hps_mem_mem_act_n[0]
set_location_assignment PIN_A24 -to hps_mem_mem_cke[0]
set_location_assignment PIN_A22 -to hps_mem_mem_cs_n[0]
set_location_assignment PIN_AG24 -to hps_mem_mem_alert_n[0]
set_location_assignment PIN_A18 -to hps_mem_mem_par[0]
set_location_assignment PIN_A19 -to hps_mem_mem_reset_n[0]
set_location_assignment PIN_B20 -to hps_mem_mem_ck[0]
set_location_assignment PIN_B19 -to hps_mem_mem_ck_n[0]
set_location_assignment PIN_A26 -to hps_mem_mem_odt[0]

set_location_assignment PIN_D18 -to hps_io_hps_io_phery_usb0_CLK
set_location_assignment PIN_C19 -to hps_io_hps_io_phery_usb0_DIR
set_location_assignment PIN_F17 -to hps_io_hps_io_phery_usb0_NXT
set_location_assignment PIN_E18 -to hps_io_hps_io_phery_usb0_STP
set_location_assignment PIN_J20 -to hps_io_hps_io_gpio_gpio1_io5
set_location_assignment PIN_N20 -to hps_io_hps_io_gpio_gpio1_io14
set_location_assignment PIN_K23 -to hps_io_hps_io_gpio_gpio1_io16
set_location_assignment PIN_L23 -to hps_io_hps_io_gpio_gpio1_io17
set_location_assignment PIN_K21 -to hps_io_hps_io_phery_emac0_MDIO
set_location_assignment PIN_K20 -to hps_io_hps_io_phery_emac0_MDC
set_location_assignment PIN_H18 -to hps_io_hps_io_phery_emac0_TX_CLK
set_location_assignment PIN_G20 -to hps_io_hps_io_phery_emac0_RXD0
set_location_assignment PIN_G21 -to hps_io_hps_io_phery_emac0_RXD1
set_location_assignment PIN_F22 -to hps_io_hps_io_phery_emac0_RXD2
set_location_assignment PIN_G22 -to hps_io_hps_io_phery_emac0_RXD3
set_location_assignment PIN_F18 -to hps_io_hps_io_phery_emac0_RX_CLK
set_location_assignment PIN_G17 -to hps_io_hps_io_phery_emac0_RX_CTL
set_location_assignment PIN_E20 -to hps_io_hps_io_phery_emac0_TXD0
set_location_assignment PIN_F20 -to hps_io_hps_io_phery_emac0_TXD1
set_location_assignment PIN_F19 -to hps_io_hps_io_phery_emac0_TXD2
set_location_assignment PIN_G19 -to hps_io_hps_io_phery_emac0_TXD3
set_location_assignment PIN_H19 -to hps_io_hps_io_phery_emac0_TX_CTL
set_location_assignment PIN_M20 -to hps_io_hps_io_phery_i2c1_SCL
set_location_assignment PIN_L20 -to hps_io_hps_io_phery_i2c1_SDA


# ......................................................................

# Set properties
# Previously we use set_global_assignment -name Fitter_Aggressive_Routability_Optimization ALWAYS 
# However, this will let the fitter to aggressively optimizes for routability. 
# Performing aggressive routability optimizations may decrease design speed, 
# but may also reduce routing wire usage and routing time. 
# The default setting of Automatically lets the fitter decide whether to 
# perform these optimizations based on the routability and timing requirements of the design

set_global_assignment -name MAX_FANOUT 20

