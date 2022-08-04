## Attendance Support - PBL6 Project

[![pipeline status](https://gitlab.com/trinhdvt/pbl6-attendace-support/badges/master/pipeline.svg)](https://gitlab.com/trinhdvt/pbl6-attendace-support/-/commits/master)

## Table of contents

* [Overview](#overview)
* [Features](#features)
* [Technologies](#techonologies)
* [Performance](#performance)
* [Demo](#demo)
* [Authors](#authors)
* [Appendix](#appendix)


# Overview

Attendance Support is a robust, AI-based system solution that will assist teachers in managing students' attendance in online exams with face comparison and student ID card identification.

By adopting state-of-the-art deep learning models in computer vision and distributed architecture, the proposed system has 98% precision in machine learning tasks, is highly durable, and is easy to scale.

When using this system, student attendance checking can be done automatically with no effort, and the needed time for one student can be reduced to around 1 second rather than 4-5 seconds when done manually.

## Features

For teachers:

- Create an exam (with a time limit for possible attendance taking)
- View all student's attendance
- Modify the record if there is any recognized information was incorrect

For students:

- Make an attendance check with provided exam code (by uploading the face's image and student id card image)

## Technologies

* Frontend: VueJS
* Backend: FastAPI, PyTorch, Celery, Redis, Express, MySQL
* Others: Docker, Docker Compose, Nginx

## Performance

The average processing time for requesting an attendance check is:

-   Server without GPU(2 Cores, 4 GB RAM): 1 second/request.

## Demo

[Demo video](https://youtu.be/viNgGmjLAK4) (Vietnamese)


## Authors

- [github@trinhdvt](https://www.github.com/trinhdvt) - Backend
- [@chauvy](#) - Frontend
- [@nghiapham](#) - Testing


## Appendix

- [Backend Repo](https://gitlab.com/trinhdvt/pbl6-attendance-sp-backend) (Java version)

- [Backend Repo](https://gitlab.com/trinhdvt/pbl6-backend-node) (NodeJS version) - After considering memory consumption when using Spring, the whole backend was rewritten to NodeJS with higher performance and lower memory usage.

- [System Deploy Documentation](https://gitlab.com/trinhdvt/pbl6-attendace-support/-/wikis/T%C3%A0i-li%E1%BB%87u-h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-tri%E1%BB%83n-khai-c%E1%BA%A5u-h%C3%ACnh-Server-cho-%C4%91%E1%BB%93-%C3%A1n-PBL6)
