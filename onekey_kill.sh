ps -ef|grep python|grep train|grep -v grep|cut -c 9-15|xargs kill -9
