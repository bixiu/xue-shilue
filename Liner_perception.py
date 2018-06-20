from numpy import *
import random
import itertools
import matplotlib.pyplot as plt
import time

class liner_perceptron(object):
	def __init__(self,train_data,rate):
		self.original_data=list(train_data)
		self.rate=rate
		self.s=train_data.shape
		self.X_label=train_data[:,-1]
		self.X=delete(train_data,-1,axis=1)
		one=ones(self.s[0])
		self.X=column_stack((self.X,one))
		self.W=ones(self.s[1])
		#self.W+=0.1
		#self.W=random.rand(1,self.s[1]+1)
	
	def __update_W(self):
		for temp in range (self.s[0]):
			if self.X_label[temp]==1:
				if float(self.W*mat(self.X[temp]).T)<0:
					continue
				elif float(self.W*mat(self.X[temp]).T)>0:
					self.W=self.W-self.rate*self.X[temp]
			else:
				if float(self.W*mat(self.X[temp]).T)>0:
					continue
				elif float(self.W*mat(self.X[temp]).T)<0:
					self.W=self.W+self.rate*self.X[temp]
		print(self.W)
	
	def line_chart(self,item):
		x=[i for i in item]
		y=[item[i] for i in item]
		y=[i/self.s[0] for i in y]
		plt.figure()
		plt.plot(x,y,linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）  
		plt.xlabel("The number of iterations") #X轴标签  
		plt.ylabel("False rate")  #Y轴标签  
		plt.show()
	
	def main(self):
		num=0
		item={}
		while True:
			F=0
			for temp in range (self.s[0]):
				if self.X_label[temp]==1:
					if float(self.W*mat(self.X[temp]).T)<0:
						continue
					elif float(self.W*mat(self.X[temp]).T)>0:
						F+=1
				else:
					if float(self.W*mat(self.X[temp]).T)>0:
						continue
					elif float(self.W*mat(self.X[temp]).T)<0:
						F+=1
			item[num]=F
			self.__update_W()
			num+=1
			if F==0:
				break
			else:
				W=self.W
		#for i in range(100):
			#self.__update_W()
		print(-float(self.W[0])/float(self.W[1]))
		print(-float(self.W[2])/float(self.W[1]))
		plt.figure()
		x1=[];x2=[];y1=[];y2=[]
		for temp in self.original_data:
			if temp[2]==1:
				x1.append(temp[0])
				y1.append(temp[1])
			else:
				x2.append(temp[0])
				y2.append(temp[1])
		ax=plt.subplot()
		ax.scatter(x1,y1,c='red')
		ax.scatter(x2,y2,c='green')
		x=arange(0,50)
		y=-float(self.W[0])/float(self.W[1])*x-float(self.W[2])/float(self.W[1])
		plt.plot(x,y)
		plt.show()
		self.line_chart(item)
		
def generate_data():
	#random_list=list(itertools.product(range(8),range(8)))
	#random_coordinate=random.sample(random_list,50)#生成随机的不重复的2维坐标
	random_coordinate=[]
	for i in range(500):
		x=random.uniform(0,50)
		y=random.uniform(0,50)
		random_coordinate.append((x,y))
	w=[1/2,10]
	#w.append(random.randint(-10,10))
	#w.append(random.randint(40,60))
	data=[]
	for i in random_coordinate:
		temp=[i[0],1]
		if w*mat(temp).T>i[1]:
			temp=list(i[:])
			temp.append(0)
			temp[1]=temp[1]+100
			temp=tuple(temp)
			data.append(temp)
		elif w*mat(temp).T<i[1]:
			temp=list(i[:])
			temp.append(1)
			temp[1]=temp[1]-100
			temp=tuple(temp)
			data.append(temp)
	'''
	plt.figure()
	x1=[];x2=[];y1=[];y2=[]
	for temp in data:
		if temp[2]==1:
			x1.append(temp[0])
			y1.append(temp[1])
		else:
			x2.append(temp[0])
			y2.append(temp[1])
	ax=plt.subplot()
	ax.scatter(x1,y1,c='red')
	ax.scatter(x2,y2,c='green')
	plt.show()
	'''
	return data

#train_data=array([(1,0,1),(0,1,1),(2,0,1),(2,2,1),(-1,-1,0),(-1,0,0),(-2,-1,0),(0,-2,0)])
train_data=generate_data()
#train_data=[(30.9754356, 29.65040756), (0.92606788, -0.74483747), (30.10885307, 30.48787502), (0.97971, -0.77278195), (30.06968994, 29.24584334), (29.39986745, 28.65523815), (0.3552677, 0.51435146), (32.99435917, 29.9140692), (0.42318196, -2.14763655), (28.7844018, 30.40698569), (-1.93254909, 0.50351952), (-1.25294209, 1.56203956), (1.16668071, -0.29767561), (31.58199401, 28.68554562), (29.90148829, 30.18892815), (0.28974657, 1.9601101), (30.32602751, 31.80295281), (29.83238713, 28.49381639), (0.07200986, 1.20929036), (-1.5582184, 2.10559211), (29.45328724, 30.1624208), (-0.04548437, 0.31533072), (-0.11912757, -0.93459693), (30.0780694, 29.68886606), (0.2146631, -0.8667782), (0.7025702, -1.51715377), (30.33314725, 30.02318205), (29.55635821, 30.15053677), (30.76408603, 28.98562219), (29.5762682, 29.21902773), (31.07619771, 30.63711598), (29.33780765, 31.1398976), (30.8107161, 29.88241233), (2.20983682, -1.01327132), (29.01290143, 29.80383821), (-0.48756727, 0.82285062), (1.29534427, -0.71874403), (30.48577719, 27.53425949), (0.9342017, -3.17833369), (-0.60079626, -0.48618088), (29.72580904, 29.88091388), (-0.26289187, 0.17151338), (29.11998248, 29.62513904), (0.37676473, -0.46177662), (30.83562039, 29.21560002), (30.37073383, 29.7212071), (0.63204951, 0.53775593), (28.63053194, 30.05330572), (1.10472266, -0.09592609), (-1.1823942, 0.29547715), (-0.9224329, 0.42490144), (0.32021556, 0.63309222), (31.07887401, 33.11323912), (31.00808474, 31.08875598), (-0.54918944, -0.91558631), (30.5807284, 30.46065377), (-0.27557961, -0.43221858), (1.86795672, 0.52970905), (31.362428, 29.84348602), (29.90637513, 29.52113816), (0.90507537, -0.21448414), (31.40718895, 29.71549982), (1.04591261, 1.41831568), (30.77321579, 31.22321955), (30.30848191, 30.22234397), (-0.69250021, -1.51949611), (31.63378444, 30.09027791), (-1.59146097, -2.39137197), (28.60975172, 30.3182253), (0.74167393, 1.12582289), (30.3178665, 29.46321458), (-0.50891635, 1.28270754), (1.89744082, 0.68496989), (28.5126328, 29.37413373), (0.31463441, 3.07556401)]
train_data=array(train_data)
#print(train_data.shape)
#label=[1.,-1.,1.,-1.,1.,1.,-1.,1.,-1.,1.,-1.,-1.,-1.,1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,1.,-1.,-1.,1.,1.,1.,1.,1.,1.,1.,-1.,1.,-1.,-1.,1.,-1.,-1.,1.,-1.,1.,-1.,1.,1.,-1.,1.,-1.,-1.,-1.,-1.,1.,1.,-1.,1.,-1.,-1.,1.,1.,-1.,1.,-1.,1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,-1.]
example1=liner_perceptron(train_data,0.01)
example2=liner_perceptron(train_data,0.07)
example3=liner_perceptron(train_data,0.3)
example4=liner_perceptron(train_data,0.7)
example1.main()
example2.main()
example3.main()
example4.main()

#generate_data()