
## Attendance Support - PBL6 Project

[![pipeline status](https://gitlab.com/trinhdvt/pbl6-attendace-support/badges/master/pipeline.svg)](https://gitlab.com/trinhdvt/pbl6-attendace-support/-/commits/master)

## Table of contents

* [Overview](#overview)
* [System](#system)
    * [Technologies](#technologies)
    * [Features](#features)
    * [Performance](#performance)
    * [Requirements](#requirements)
* [Demo](#demo)
* [Status](#status)
* [Contact](#contact)
* [Appendix](#appendix)

## Overview

Attendance Support  is my AI's project which is developed to help teachers in student attendance check with face comparison and student's id card identification.

## System

### Technologies

* Image Processing with **[OpenCV](https://github.com/opencv/opencv-python)**
* Object Detection with **[YOLOv5](https://github.com/ultralytics/yolov5)**
* Text Recognition with **[Transformer](https://github.com/pbcquoc/vietocr)**
* **[FastAPI](https://github.com/tiangolo/fastapi)** for create API
* **[Celery](https://github.com/celery/celery)** and **[Redis](https://redis.io/)** for create **Message Queue Architecture**
* **[Docker](https://www.docker.com/)** for deployment

### Features

Current features:

* The main function of the system is to identify and extract information from Student ID Card photos as well as compare student uploaded face with extracted one from ID Card. 
* Student can upload a card and a face image to the system, and this system will return extracted information from the photo, for example:
    * ID Number
    * Full name
    * Class name
    * Date of birth
    * Faculty
    * Year
    * Match (**True** if both faces match, otherwise **No**)

To-do list:

- [ ] Use an Object Detection model (YOLO, SSD, etc) for scanning ID Card in the input image instead of using Image Processing.
- [ ] Add one more function to this system: Vietnamese ID Card Recognition
- [ ] Create a better UI Website


### Performance

Because I don't have a server with GPU, so the processing time is quite long. The average processing time is:

* Server without GPU(2 Cores, 4 GB RAM): 1 - 2 seconds/request.

### Requirements

* The input card image must have 4 clear angles and its background should be white or gray and not contain anything else to make sure the ID Card can be *seen*.
* The input face image must have a face inside.
* All information fields must be visible, readable, unmodified, and not blurred.
* Both input image size does not exceed **6 MB**, and the minimum resolution is approximately **640x480** to ensure the confident rate.
* The ratio of Student ID Card area must be at least 2⁄3 of the total image area.

## Demo

> Currently, I'm building a demo website to show how it works. Please stay tune!

> ~~I have built a simple **[website](#)** to show how it works. However, to use this website you need a Student ID Card as your input image. So, please download the [test-img folder](https://drive.google.com/drive/folders/1tOklpJxGfGlmfr4Ui1fSAEN_lPZVU5H_?usp=sharing) then use one of them. Sorry for this inconvenience~~ 

## Status

The project is still continuously under my development. I'm trying to create a nice UI for the system and refactor code with much more cleaner.

## Contact

Created by [@dvt](https://www.facebook.com/trinh.dvt/) \- feel free to contact me\!

## Appendix

[System Deploy Documentation](https://gitlab.com/trinhdvt/pbl6-attendace-support/-/wikis/T%C3%A0i-li%E1%BB%87u-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-tri%E1%BB%83n-khai-c%E1%BA%A5u-h%C3%ACnh-Server-cho-%C4%91%E1%BB%93-%C3%A1n-PBL6)
