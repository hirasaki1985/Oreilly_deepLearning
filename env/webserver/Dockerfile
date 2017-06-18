FROM docker.io/nginx
MAINTAINER mamoru hirasaki

######################
# init
######################
RUN yum -y install epel-release
RUN yum -y update kernel
RUN yum -y install kernel-devel kernel-headers dkms gcc gcc-c++
RUN yum -y install http://vault.centos.org/6.5/updates/x86_64/Packages/kernel-devel-2.6.32-431.3.1.el6.x86_64.rpm
RUN yum -y install vim, lsof, which, net-tools, wget.x86_64 curl

######################
# install
######################
# python
RUN yum install -y python-pip
RUN yum install -y gcc zlib-devel bzip2 bzip2-devel readline readline-devel sqlite sqlite-devel openssl openssl-devel git patch
RUN git clone https://github.com/yyuu/pyenv.git ~/.pyenv

# git
# git
WORKDIR /usr/local/nginx
RUN git clone https://github.com/hirasaki1985/Oreilly_deepLearning04web_webserver.git
#RUN cp -p simpleserver.py /usr/share/nginx/html/index.html

# update
RUN yum -y update

######################
# start
######################
RUN service nginx start
