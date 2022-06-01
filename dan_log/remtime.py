def printTime(remtime):
	# 把剩余时间转化我时分秒
	hrs = int(remtime)/3600
	mins = int((remtime/60-hrs*60))
	secs = int(remtime-mins*60-hrs*3600)
	timedisp="Time remaining : "
	if hrs>0:
		timedisp+=str(hrs)+"Hrs "
	if mins>0:
		timedisp+=str(mins)+"Mins "
	timedisp += str(secs)+"Secs"
	print(timedisp)
