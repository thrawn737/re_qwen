UW Sinus Surgery Cadaver/Live (UW-Sinus-Surgery-C/L) Dataset:

[Dataset Description]
This dataset was developed at the University of Washington's BioRobotics Lab (http://brl.ee.washington.edu). It has endoscopic sinus surgery images with manual annotations for surgical instrument segmentation task. The challenging conditions of this dataset include specular reflections, blur from motion, blood, smoke and tools in shadow.

The images were extracted from 10 cadaveric surgery videos and 3 live surgery videos. The videos were collected using a Stryker 1088 HD camera system and the Karl Storz Hopkins 4mm 0° endoscope.

The dataset is arranged as follows:
|--- readme.txt
|--- cadaver
|     |---images
|     |---labels
|     
|—— live
      |---images
      |---labels
UW-Sinus-Surgery-C/L consists of two datasets: cadaver dataset (UW-Sinus-Surgery-C) and live dataset (UW-Sinus-Surgery-L). In each dataset, folder "images" has surgery images and folder "labels" has segmentation ground truths of the corresponding surgery images. The segmentation ground truths are given by images whose pixel values indicate the class index. Here 0 stands for background and 1 for surgical instrument.

Image naming:
i) Sinus-Surgery-C dataset: S[video_ID]_[frame_index]
ii) Sinus-Surgery-L dataset: L[video_ID]_[frame_index]
[video_ID] shows from which video the image was extracted, [frame_index] is the frame's index in the corresponding video.

3-fold cross-validation experimental setup: 
In [2], the image segmentation performances were evaluated based on the K-fold cross-validation method, to avoid the bias caused by dataset split. The 3-fold cross-validation is conducted with the setup below:
i) Sinus-Surgery-C dataset: fold 1: 1st~4th videos; fold 2: 5th-7th videos; fold 3: 8th-10th videos
ii) Sinus-Surgery-L dataset: fold 1: 1st video; fold 2: 2nd video; fold 3: 3rd video

[Contact] Shan Lin (shanl3@uw.edu), Fangbo Qin (fbqin@uw.edu)

If you find this dataset helpful in your research, please cite our papers [1] and [2]. For any suggestions or corrections, please contact us at shanl3@uw.edu or fbqin@uw.edu. 

[References]
[1] S. Lin, F. Qin, Y. Li, et al., “LC-GAN: Image-to-image translation based on generative adversarial network for endoscopic images,” arXiv preprint arXiv:2003.04949, 2020. (accepted by IROS 2020)
[2] F. Qin, S. Lin, Y. Li, et al., “Towards better surgical instrument segmentation in endoscopic vision: multi-angle feature aggregation and contour supervision,” IEEE Robotics and Automation Letters, 2020 (IROS2020 presentation), https://ieeexplore.ieee.org/document/9140341
[3] S. Lin, F. Qin, R. A. Bly, et al., "Automatic Sinus Surgery Skill Assessment Based on Instrument Segmentation and Tracking in Endoscopic Video," in Proc. Int. Workshop Multiscale Multimodal Med. Imag., pp.93-100, 2019.
[4] S. Lin, X. Gu, R. A. Bly, et al., "Video-based automatic and objective endoscopic sinus surgery skill assessment," in Medical Imaging 2020: Image-Guided Procedures, Robotic Interventions, and Modeling., vol. 11315, p. 113152L, 2020.