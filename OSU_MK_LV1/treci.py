list = []
while 1:
    word = input()
    if word.isdigit():
        list.append(int(word))
    elif word == "Done":
        break
    else:
        print("NaN")
        continue
    
numbers = sorted(list)
print(f"Uneseno brojeva: {len(numbers)}")
print(f"Sortirana lista: {numbers}")
print(f"Minimalna vrijednost u listi: {min(numbers)}")
print(f"Maksimalna vrijednost u listi: {max(numbers)}")
print(f"Srednja vrijednost: {sum(numbers)/len(numbers)}")
