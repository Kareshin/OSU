ham_count=0
spam_count=0
ham_words=0
spam_words=0
exCount = 0

fhand = open ('SMSSpamCollection.txt')

for line in fhand :
    line = line.rstrip()
    words = line.split()
    if words[0]=="ham":
        ham_count+=1
        ham_words+=len(words[1:])
    else:
        spam_count+=1
        spam_words+=len(words[1:])
        if line.endswith('!'):
            exCount+=1
fhand.close()

print(f"Prosjecan broj rijeci u ham porukama: {ham_words/ham_count}")
print(f"Prosjecan broj rijeci u spam porukama: {spam_words/spam_count}")
print(f"Broj spam poruka koje zavrsavaju usklicnikom: {exCount}")
