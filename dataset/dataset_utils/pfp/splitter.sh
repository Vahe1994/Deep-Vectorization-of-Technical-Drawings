#!/usr/bin/env bash
test_samples=('0018tcr22ekdb6mgd.svg' '00we6a11n604j7kplorg8.svg' '01e0jdfbkedr4aw8gd.svg' '029oce6m4jgfe1il1d.svg' '02khep9768mme7o7dlti0.svg' '031lannjs0niuwd.svg' '03otrk34angwlgejd.svg' '03pr64nn74j59bdd.svg' '06s5p8nnf2358d1wnmd.svg' '077ju07176jeaomprh7d.svg' '07lt9j7i9b7rk9uk5.svg' '085wep1s7w46trj23d.svg' '08ak3hbcb297i820kmtd.svg' '08ckbbdjmht7cm2por0.svg' '0afnwd5t2gslfi5ol4.svg' '0al1uoe8k9r3ad.svg' '0b3135acauhdg8hp6l2.svg' '0b3t73252e292tetd.svg' '0ca46nltn4o89gf13.svg' '0ckboi08n4pt285d.svg' '0cnh8ihf3041wuad.svg' '0d1ho7mkh6wopu4mfb5.svg' '0dw2nf8ai8cw21b0ud.svg' '0e2osb5fpoe252bfa9m9.svg' '0eabecc6l9b7upak74d.svg' '0f7fg5bpgar4g0lld.svg' '0ha4gspuna9ou0c7oi8.svg' '0hojue3j7ui7w8iand.svg' '0i8igfw4tnbu4n6bj0.svg' '0j4bwcr578m6rpg4j35.svg' '0j4n8mg0ec5iuofuf6d.svg' '0je59lh49edt9eb8hnd.svg' '0jmc30rilein6p566d.svg' '0kbuoe5u1tan317.svg' '0ld367w5f6o7m9cll09.svg' '0lir5m95u7c62nurj2.svg' '0m3ljblw88pb5o5o37dw2.svg' '0m401j1r9p93u3pm25sc0.svg' '0n6bgh943udahag8d.svg' '0nmouo97uun390ehgoj32.svg')
val_samples_n=150

cd /data/svg_datasets/whole_images/precision-floorplan

# all to train
mkdir -p train
for f in all/*.svg
do
	ln "$f" train/
done

# test_samples to test
mkdir -p test
for f in "${test_samples[@]}"
do
	mv "train/$f" test/
done

# split train / val
mkdir -p val
val_samples_n_so_far=0
for f in train/*.svg
do
	mv "$f" val/
	val_samples_n_so_far=$((val_samples_n_so_far+1))
	if [ "$val_samples_n_so_far" -eq "$val_samples_n" ]; then break; fi
done
