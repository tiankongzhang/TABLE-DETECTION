#开始检测
python3 eval.py --test_data_path=data_new --checkpoint_path=checkpoint/detect --output_dir=data_new 
#开始识别
python3 ocrReg.py --test_data_path=data_new --reg_data_path=Result
