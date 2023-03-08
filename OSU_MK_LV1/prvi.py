
def total_euro(radniSati, euroh):
    return radniSati*euroh


radniSati=float(input("Radni sati: "))

euroh=float(input("Euro/sat: "))



print("Ukupno: {}" .format(total_euro(radniSati,euroh)))