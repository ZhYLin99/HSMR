config=configs/whole_body.yml
log_dir=bot_r18_extract_mot_features_whole_body
model_dir=bot_r18_train_on_iros2022_fisheye_whole_body/model_best.pth

videos=(
#  02_black_black_fisheye_head_front
  02_blue_black_fisheye_head_front
  02_color_gray_fisheye_head_front
  02_original_black_fisheye_head_front
  02_suit_black_fisheye_head_front
  06_color_gray_fisheye_head_front
  06_gray_gray_fisheye_head_front
  06_original_black_fisheye_head_front
  06_red_gray_fisheye_head_front
  06_suit_black_fisheye_head_front
  10_black_black_fisheye_head_front
  10_blue_black_fisheye_head_front
  10_gray_gray_fisheye_head_front
  10_original_black_fisheye_head_front
  10_red_gray_fisheye_head_front
  12_blue_black_fisheye_head_front
  12_color_gray_fisheye_head_front
  12_original_black_fisheye_head_front
  12_suit_black_fisheye_head_front
  12_white_black_fisheye_head_front
  14_color_gray_fisheye_head_front
  14_gray_gray_fisheye_head_front
  14_original_black_fisheye_head_front
  14_red_gray_fisheye_head_front
  14_suit_black_fisheye_head_front
  16_blue_black_fisheye_head_front
  16_color_black_fisheye_head_front
  16_original_black_fisheye_head_front
  16_red_gray_fisheye_head_front
  16_suit_black_fisheye_head_front
  18_blue_black_fisheye_head_front
  18_gray_gray_fisheye_head_front
  18_original_black_fisheye_head_front
  18_red_gray_fisheye_head_front
  18_suit_black_fisheye_head_front
  20_gray_gray_fisheye_head_front
  20_original_black_fisheye_head_front
  20_red_gray_fisheye_head_front
  20_suit_black_fisheye_head_front
  20_white_black_fisheye_head_front
)

  for video in ${videos[*]}
do

  {
    if [[ $video == *"original"* ]]
    then
      seq=../../../../tracking/eval/data/gt/zjlab/iros2022-fisheye-original-test/${video}
    else
      seq=../../../../tracking/eval/data/gt/zjlab/iros2022-fisheye-similar-test/${video}
    fi
    rm ${seq}/embedding/embedding_wb.pkl
    rm ${seq}/embedding/embedding_hs.pkl

  }&
done