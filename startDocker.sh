#!/bin/bash
docker build -t mypython docker/ && docker container run -it -p 8888:8888 -v `pwd`:/data mypython

