import numpy as np
import matplotlib . pyplot as plt

arr=np.genfromtxt("data.csv",delimiter=",")

arr=np.delete(arr,0,axis=0)


rows, cols=np.shape(arr)
print("Mjerenja su izvrsena na "+str(rows)+" osoba")

x=arr[:,1] #height
y=arr[:,2] #weight


plt.scatter(x,y)
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Odnos visine i mase")
plt.show()

x5=x[::50]
y5=y[::50]

plt.scatter(x5,y5)
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Odnos visine i mase svakih 50 osoba")
plt.show()

print("Minimalna vrijednost visine je: "+str(np.min(x)))
print("Maksimalna vrijednost visine je: "+str(np.max(x)))
print("Srednja vrijednost visine je: "+str(np.mean(x)))


index=(arr[:,0]==1)
heightMen=[]





