FROM docker.io/centos
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

######################
# setting
######################
# python
RUN echo '# pyenv' >> ~/.bash_profile
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
RUN echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
RUN source ~/.bash_profile
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

RUN pyenv install anaconda3-4.2.0
#RUN pyenv install 3.5.2
RUN pyenv global anaconda3-4.2.0

# git
RUN mkdir /develop
RUN mkdir /develop/logs
RUN chmod 777 /develop
RUN chmod 777 /develop/logs
WORKDIR /develop
RUN git clone https://github.com/hirasaki1985/Oreilly_deepLearning04web_appserver.git
WORKDIR /develop/Oreilly_deepLearning_04web
#RUN cp -p simpleserver.py /usr/share/nginx/html/index.html

# update
RUN yum -y update

######################
# start
######################
# pythonweb
EXPOSE 8000
RUN nohup python -u -m simpleserver.py 8000 > ~/logs/pythonweb.log &

CMD ["/bin/bash"]

######################
#sudo docker run --privileged -td --name Oreilly_deepLearning_web docker.io/centos
#sudo docker start Oreilly_deepLearning_web
#sudo docker exec -it Oreilly_deepLearning_web /bin/bash
######################

