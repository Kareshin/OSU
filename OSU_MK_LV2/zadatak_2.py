import numpy as np
import matplotlib . pyplot as plt

arr=np.genfromtxt("data.csv",delimiter=",")

arr=np.delete(arr,0,axis=0)

#a
rows, cols=np.shape(arr)
print("Mjerenja su izvrsena na "+str(rows)+" osoba")

#b

x=arr[:,1] #height (znaci u x se sprema samo vrijednosti prvog stupca)
y=arr[:,2] #weight (znaci u y se sprema samo vrijednosti drugog stupca)


plt.scatter(x,y)
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Odnos visine i mase")
plt.show()

#c
x5=x[::50] #korak 50, svaki 50-ti
y5=y[::50]

plt.scatter(x5,y5)
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Odnos visine i mase svakih 50 osoba")
plt.show()

#d
print("Minimalna vrijednost visine je: "+str(np.min(x)))
print("Maksimalna vrijednost visine je: "+str(np.max(x)))
print("Srednja vrijednost visine je: "+str(np.mean(x)))

#e
male = []
female = []
for people in arr[::50]:
    if people[0] == 1:
        male.append(people[1])
    else:
        female.append(people[1])
male = np.array(male)
female = np.array(female)
print("Minimalna visina muskaraca: ", male.min())
print("Maksimalna visina muskaraca: ", male.max())
print("Prosjecna visina muskaraca: ", male.mean())
print("\n")

print("Minimalna visina zena: ", female.min())
print("Maksimalna visina zena: ", female.max())
print("Prosjecna visina zena: ", female.mean())
