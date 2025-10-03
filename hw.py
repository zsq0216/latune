
a,b,c=input().split()
a=float(a)
b=float(b)
c=float(c)
x1 = (-b + (b*b-4*a*c)**(1/2)/(2*a))
x2 = (-b - (b*b-4*a*c)**(1/2)/(2*a))
if b**2 == 4 * a * c:
    print("x1=x2="+str(x1))
if b**2 > 4 * a * c:
    print("x1="+str(x1), "x2="+str(x2))
if b**2 < 4 * a * c:
    print("x1="+str("%.5f"%(-b/(2*a)))+'+'+str("%.5f"%(((4*a*c-b*b)**(1/2))/(2*a)))+'i',"x2="+str("%.5f"%(-b/(2*a)))+'-'+str("%.5f"%(((4*a*c-b*b)**(1/2))/(2*a)))+'i')
