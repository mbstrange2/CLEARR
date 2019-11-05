N = 10000;
alpha = 2;
fnyq = 20e3;
Vdd = 1;
Cline = 900e-15;
vn2 = (10e-6)^2;
gm_ID = 25;
dc = 0.5;
B = 8;
kt = 300*1.38e-23;

Pdriver = Cline*Vdd^2*fnyq*alpha/N
Pramp = (sqrt(N)+1)*Cline*Vdd^2*fnyq/N
Pbox = 4*kt/vn2/gm_ID*dc*fnyq
Pck = (1+sqrt(N))*fnyq*2^B*Cline*Vdd^2/N