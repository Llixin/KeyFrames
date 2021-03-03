# 代表帧算法说明

## 1 介绍

### 1.1 解决的问题

找出一段视频的代表帧

### 1.2 算法原理

根据NIMA算法选取得分最高的一些帧，然后去除含模糊、闭眼、侧脸的不良图片，最终得到视频的代表帧的候选集合

### 1.3 算法消耗的资源

2000M显存

## 2 算法依赖环境配置

### 2.1 运行环境

- Ubuntu18.04 LTS
- Python3.6
- CUDA >= 10.0

### 2.2 依赖安装

    pip install -r requirement.txt

 
## 3 算法使用说明

### 3.1 算法调用

    from keyframes.keyframes import KeyFrames      
                                                   
    video_path = "video/gdxw/gdxw_20180419.mp4"
    json_path = "video/gdxw/gdxw_20180419.json"
    
    test = KeyFrames()
    kf_im = test.predict(video_path, json_path)
    test.save_img(kf_im, "video/gdxw/result/")
    

## 4 注意事项
描述下在使用算法过程中需要注意的一些点
1.导入本算法的时候，注意路径问题

    from keyframes.keyframes import KeyFrames
    
2.python版本
    在python3.6.9和python3.8.2两个版本测试过
    这两个版本之间的版本应该都没有问题，若要使用其他版本的python，请自行尝试