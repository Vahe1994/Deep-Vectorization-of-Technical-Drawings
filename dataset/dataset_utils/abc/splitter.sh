#!/usr/bin/env bash
monsters=('00055154_edbb80ecd3474f4ca2fd2500_step_020.svg' '00051830_8810b2df7f8548f7bebe761b_step_000.svg' '00054539_4cfbc1200edc4a47a5e924f7_step_000.svg' '00051817_eaf5921a94434133971338e9_step_000.svg' '00051310_a3cf41ca7a9842d19ea467f2_step_001.svg' '00053154_2f0af6e099aa47599fe6ed8d_step_000.svg' '00058830_be180691f57e4f1fbf0caec3_step_004.svg')
test_samples=('00050000_99fd5beca7714bc586260b6a_step_000.svg' '00050001_82639ddcab054f4eb6d953b3_step_000.svg' '00050002_790983e52c4e4efea4148e97_step_000.svg' '00050003_790983e52c4e4efea4148e97_step_001.svg' '00050007_c43496ca6f5a43fd9436097b_step_000.svg' '00050009_f5e95c39e7f941fdad8c83e5_step_000.svg' '00050012_e408ebe315f24b71a82d3868_step_000.svg' '00050014_f24ee4bf82e34a66bb54c43b_step_000.svg' '00050016_f24ee4bf82e34a66bb54c43b_step_002.svg' '00050018_7704b2cc9e024aec819857d0_step_000.svg' '00050019_54681d094f4c4ee1b4932d57_step_000.svg' '00050020_b591e7b827b045b5b65304e5_step_000.svg' '00050022_3ef62c8f5ce247168e9a1692_step_000.svg' '00050023_2fbe677818504ef79ccae595_step_000.svg' '00050024_06b49ffdd02c43c2b8c21746_step_000.svg' '00050026_825efaba00314af8b54b7397_step_000.svg' '00050031_8ada6b112790479887a6ab35_step_000.svg' '00050032_d0eaf6069a7c4435b69edcd4_step_000.svg' '00050033_d0eaf6069a7c4435b69edcd4_step_001.svg' '00050034_d0eaf6069a7c4435b69edcd4_step_002.svg' '00050035_d0eaf6069a7c4435b69edcd4_step_003.svg' '00050036_d0eaf6069a7c4435b69edcd4_step_004.svg' '00050037_d0eaf6069a7c4435b69edcd4_step_005.svg' '00050042_7034d032759540a0a3b1f58a_step_001.svg' '00050048_c5be5ed6a09b419db7e76eee_step_000.svg' '00050049_c5be5ed6a09b419db7e76eee_step_001.svg' '00050050_5c52e33ffa3d4f5eb58b81c3_step_000.svg' '00050052_5dee25046678432abb27eff8_step_000.svg' '00050053_78888aa0b452466f959d03cd_step_000.svg' '00050054_b055cfd27d714441b8a26ca5_step_000.svg' '00050055_3ab4ef952f4f490c86d0f35e_step_000.svg' '00050056_b75b54fa9fb24906ae5fb2f6_step_000.svg' '00050058_c9bd21300b25465a9250c00f_step_000.svg' '00050061_737486f5bdb54865b003b8fc_step_000.svg' '00050062_130600a01c8544bba3059b3c_step_000.svg' '00050063_d57c16e4a4a84b6a8b6fae7b_step_000.svg' '00050064_d57c16e4a4a84b6a8b6fae7b_step_001.svg' '00050069_d86c57e3d2264f41ac027e43_step_000.svg' '00050070_3fb5d5c7a57e489b8079e438_step_000.svg' '00050072_665f5589058b486a9f274b56_step_000.svg' '00050073_94addb3a638f4e22b50dc081_step_000.svg' '00050075_fd5409d35f3b4f858041c4c7_step_000.svg' '00050077_7a331cce71864354b7c51210_step_000.svg' '00050078_0afd184308784343ad0b9a50_step_000.svg' '00050080_efa49aa99ebf43e1957a6011_step_000.svg' '00050081_efa49aa99ebf43e1957a6011_step_001.svg' '00050082_efa49aa99ebf43e1957a6011_step_002.svg' '00050083_efa49aa99ebf43e1957a6011_step_003.svg' '00050084_efa49aa99ebf43e1957a6011_step_004.svg' '00050085_b1d1786087a44ab082d0ad50_step_000.svg')
val_samples_n=1500

root_dir='/data/svg_datasets/whole_images/abc'

#mkdir -p "${root_dir}/monsters"
#for i in "${monsters[@]}"; do
#    mv "${root_dir}/train/$i" "${root_dir}/monsters/"
#done

# then move duplicates first

mkdir -p "${root_dir}/test"
for i in "${test_samples[@]}"; do
    mv "${root_dir}/train/$i" "${root_dir}/test/"
done

mkdir -p "${root_dir}/val"
val_samples_n_so_far=0
for f in ${root_dir}/train/*.svg
do
	mv "$f" "${root_dir}/val/"
	val_samples_n_so_far=$((val_samples_n_so_far+1))
	if [ "$val_samples_n_so_far" -eq "$val_samples_n" ]; then break; fi
done
