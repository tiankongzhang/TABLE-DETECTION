#开始检测
nohup python3 -u ocrRegCluster_final.py --test_data_path=data/images5 --checkpoint_path=checkpoint/detect --output_chart_path=Chart --output_nochart_path=NoChart --out_detect_path=Detect --output_recog_path=Recognition.txt --output_chart_origin=chart --output_chart_view=view --out_detect_cut=cut --out_detect_view=view > extract_detect.out 2>&1 &
