## Dual-Stream Reciprocal Disentanglement Learning for Domain Adaption Person Re-Identification

### Usage
- This project is based on the strong person re-identification baseline: Bag of Tricks[3] ([paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)).
- This package contains the source code which is associated with the following paper:
```
Huafeng Li, Kaixiong Xu, Jinxing Li, et al. Dual-stream reciprocal disentanglement learning for domain adaptation person re-identification[J]. Knowledge-Based Systems, 2022, 251: 109315.
```
- Usage of this code is free for research purposes only. 

### Installation
- The model is learned by pytorch. See Bag of Tricks[3] for more Settings.

### Prepare Datasets
- Download the original dataset from [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)[1] and [MSMT17](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf)[2] and [DukeMTMC-reID](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf)[4,5].

### Train.  
(1)Please replace dataset path with your own path, see 'defaults.py' for more details.    
(2)To begin training.(See the code for more details)
```
|____ tools/
     |____ train.py/
```  
```
python train.py
```
### Test.   
(1)To begin testing.(See the code for more details)  
```
|____ tools/
     |____ test.py/
```  
```
python test.py
```
(2)Note that the proposed new protocol for datasets is used for the training set of the target domain.  
(3)You can downloading the parameter files trained in this paper.( Useing to verify the effectiveness of the proposed method).[Google Drive](https://drive.google.com/drive/folders/1KBUeVGO3WYPVf-ZXdhRObN5UJEyShR3W?usp=sharing) (or  [Baidu Disk](https://pan.baidu.com/s/1wptgchIhqqxI7tg2cSeT7Q) password:1111 for Duke2Market, Market2Duke and the proposed new protocol).

### Dataset in This Paper ( It is for research purposes only. )

- Download the original dataset from [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)[1] and [MSMT17](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf)[2].
- The new protocol of the dataset in this paper:Market-new and MSMT17-new.  
**Describe**:For Market-new, we use the training data of Market1501 to simulate the actual scene where pedestrians enter the next camera from the camera at one intersection on a straight road. Assume that six cameras are arranged at the intersection of a continuous road. All pedestrians walk up, right, down, and left to enter the next camera from one intersection as the starting point. Assume that the number of pedestrians leaving the camera in each direction is a quarter of the total number of pedestrians at the intersection (because there are four directions to choose from), that is, the number of pedestrians under the camera at that intersection is a quarter of the total number of pedestrians at the next adjacent intersection. By following the assumption mentioned above,the Training set in Market1501 under the new protocol contains 3197 images with 617 identities. The Test set is consistent with the original protocol of Market1501. According to this new setup, on straight roads, there will be a large number of isolated pedestrians under each camera. For MSMT17-new, since the testing set contains more identities than that in the training set, we inversely regard the original testing set as the training set, while the original training set is regarded as the testing set. Similarly, it is assumed that the number of pedestrians entering another adjacent camera from one camera accounts for 25 per cent of the total number of people on the camera. Under the new protocol, the msmt17 training set contains 15356 images of 1790 pedestrians, and the new test set contains 32621 images of 1041 pedestrians, in which probe contains 2900 images and gallery contains 29721 images. Please refer to the [paper](https://arxiv.org/pdf/2106.13929.pdf) for more details.
- Download the new protocol of the dataset in this paper:Market-new[Google Drive](https://drive.google.com/file/d/1hxrwLuW6jyP91Lxl4xXIJD7N63ggh1XK/view?usp=sharing) (or  [Baidu Disk](https://pan.baidu.com/s/1I0IIeZKm38V2PFi3VRmYUg) password:1111) , MSMT17-new[Google Drive](https://drive.google.com/file/d/1jnLk1RbgWWqYFHFdqRVtbn76i8dgRLI5/view?usp=sharing) (or  [Baidu Disk](https://pan.baidu.com/s/1I0IIeZKm38V2PFi3VRmYUg) password:1111) and Duke-new[Google Drive](https://drive.google.com/file/d/1Ec3KaKdDndUeJ5WgosijL9n9SADWtroB/view?usp=share_link
).  
- Please cite this paper if it helps your research:  
```
@article{li2022dual,
  title={Dual-stream reciprocal disentanglement learning for domain adaptation person re-identification},
  author={Li, Huafeng and Xu, Kaixiong and Li, Jinxing and Yu, Zhengtao},
  journal={Knowledge-Based Systems},
  volume={251},
  pages={109315},
  year={2022},
  publisher={Elsevier}
}
```
### Contact
- If you have any questions, please feel free to contact me. [xukaixiong@stu.kust.edu.cn](xukaixiong@stu.kust.edu.cn) . 

### Reference
```
[1]Zheng L, Shen L, Tian L, et al. Scalable person re-identification: A benchmark[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1116-1124.  
[2]Wei L, Zhang S, Gao W, et al. Person transfer gan to bridge domain gap for person re-identification[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 79-88.
[3]Luo H, Gu Y, Liao X, et al. Bag of tricks and a strong baseline for deep person re-identification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019: 0-0.
[4]Zhedong Zheng, Liang Zheng, Yi Yang, Unlabeled samples generated by gan improve the person re-identification baseline in vitro, in: The IEEE International Conference on Computer Vision (ICCV), 2017. pp:3754-3762.
[5]Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, Carlo Tomasi, Performance measures and a data set for multi-target, multi-camera track-ing, in: European Conference on Computer Vision Workshops (ECCVW), 2016, pp. 17-35.
```
