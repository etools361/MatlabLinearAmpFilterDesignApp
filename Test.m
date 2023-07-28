% filterDesign 
% strNetlist = {
% 'V0 V 9 0 1.000000e+00';
% 'RS R 1 9 1.000000e+00';
% 'C1 C 1 0 2.101';
% 'L2 L 1 2 1.065';
% 'C3 C 2 0 2.834';
% 'L4 L 2 3 0.789968';
% 'RL R 3 0 1.000000e+00';
% };
% % RF-Tool 
% strNetlist = {
% 'V0 V 9 0 1.000000e+00';
% 'RS R 1 9 1.000000e+00';
% 'C1 C 1 0 1.569';
% 'L2 L 1 2 1.555';
% 'C3 C 2 0 1.555';
% 'L4 L 2 3 1.569';
% 'RL R 3 0 1.000000e+00';
% };

% strNetlist = {
% 'V0 V 9 0 1.000000e+00';
% 'RS R 1 9 1.000000e+00';
% 'C1 C 1 0 2.26';
% 'L2 L 1 2 1.51';
% 'C3 C 2 0 1.51';
% 'L4 L 2 3 2.26';
% 'RL R 3 0 1.000000e+00';
% };
% 
% x=[5.156,2.281,3.771,1];
% y=[11.656,5.156,11.937,3.771,1];
% x*2.261
% 
% 
% 
% x=[3.410769,1.51, 1]

% [4.644,1.348,3.231,0.414];%)[16,4.644,           17.348,          3.231,           2.414];( 3.44530577088717*s
% [16,4.64427217915591,11.1317829457364,1.42635658914729,0];
% [0, 0, 6.2162170542636, 1.80464341085271, 2.414];%)[4.644,1.348,3.231,0.414](0.747078160151238*s
% [4.644,1.3482096791089,1.80344667860509,0]
% [0,0,1.42755332139491,0.414])[6.2162170542636, 1.80464341085271, 2.414](4.35445524948205*s
% [6.2162170542636,1.80274447328557,0]
% [2.414])[1.42755332139491,0.414](0.591364259069971*s
% [1.42755332139491,0]
% 0.414

% f=logspace(-3, 2, 10000);
% w = 2.*pi.*f;
% s = 1i.*w;
% % K = ((-0.2372 - 0.9952*1i)*(-2.2065)*(-0.2372 + 0.9952*1i))/((1.1547*1i)*(-1.1547*1i));
% % Hs = K.*(s-(1.1547*1i)).*(s-(-1.1547*1i))./((s-(-0.2372 - 0.9952*1i)).*(s-(-2.2065)).*(s-(-0.2372 + 0.9952*1i)));
% % K = ((-0.2372 - 0.9952*1i)*(-2.2065)*(-0.2372 + 0.9952*1i))/((1.1547*1i)*(-1.1547*1i)*1.6331e+16*1i);
% % Hs = K.*(s-(1.1547*1i)).*(s-(-1.1547*1i)).*(s-1.6331e+16)./((s-(-0.2372 - 0.9952*1i)).*(s-(-2.2065)).*(s-(-0.2372 + 0.9952*1i)));
% Hs = (s.^2+1.33)./(s.^3+2.6808.*s.^2+2.0934.*s+2.3095);
% Hs_dB = 20*log10(abs(Hs));
% semilogx(f, Hs_dB, '-r');
% grid on;

% x^2 + 1.33333
% s^3 + 2.6809 s^2 + 2.09345 s + 2.30951


% (0.65 x^2 + 1)+0.8438 x * (0.4219 x)       x^2 + 1
% -----------------------------        = ---------------------------
% (0.65 x^2 + 1)*(0.4219 x)               0.274235 x (x^2 + 1.53846)
% 
% x^2 + 1                                                   x^2 + 1
% -----------------------------------------------   = ----------------------------
% (x^2 + 1)*0.4219 x + 0.274235 x (x^2 + 1.53846)      0.696135 x^3 + 0.8438 x


% ( x^2 + 0.9941)/(0.6945 x^3 + 0.8388 x)
% n=3;
% Rs = 1;
% Ws  = 0.1710;
% [b,a] = cheby2(n,Rs,Ws);
% NN = 100;
% a=linspace(-2,2,NN);%real
% b=linspace(-2,2,NN);%image
% [A,B]=meshgrid(a,b);
% % s2 = 1i.*b(b>=0);
% s=A+1i*B;
% 
% % 1st RC
% R = 1;
% C = 1;
% H = 1./(s.*R.*C+1);
% 
% H_abs = abs(H);
% surf(A, B, H_abs);
% grid on;
% N  = 100;
% phi = linspace(0, 180, N);
% for ii=1:N
%     a = phi(ii)./180*pi;
%     x0 = 4;
%     x = [1:0.1:10];
%     z = sin(x);
%     xp = (x-x0).*cos(a)+x0;
%     zp = z;
%     plot(xp, zp);
%     xlim([-max(x), max(x)]);
%     drawnow;
%     pause(0.05);
% end


% Ap = 3;
% As = 50;
% fs = 1;
% n  = 4;
% 
% es       = sqrt(10^(0.1*As)-1);
% ep       = sqrt(10^(0.1*Ap)-1);
% k1       = ep/es;
% k        = ellipdeg(n, k1);
% 
% v2   = (n-1)/(n);
% wa   = cde(v2, k);
% wb   = 1./(k*cde((n-1)/n, k));
% aa   = wb^2;
% dd   = (-1+wb^2)/(1-wa^2);
% bb   = dd*wa^2;
% cc   = 1;
% 
% 
% [K,Kp]   = ellipke(k);
% [K1,K1p] = ellipke(k1);
% Nexact   = (K1p/K1)/(Kp/K);
% W        = logspace(log10(0.1), log10(10), 1000);
% % W  = linspace((0.1), (3), 1000);
% % Phi = delta.*sne(n.*ellipk(delta)./ellipk(k).*asne(W./sqrt(k), k), delta^2);
% % Lk = cde(n.*K1./K.*acde(1/k, k), k1);
% % Lk = 1/abs(Lk)
% % [K1,K1p] = ellipke(Lk);
% if mod(n, 2)
%     W0 = W;
% else
%     W0 =sqrt((aa.*W.^2+bb)./(cc.*W.^2+dd));
%     W1 =sqrt(W.^2.*(1-wa.^2)+wa.^2);
% end
% Phi2 = ep.*cos(n.*acos(W1));
% 
% Phi  = ep.*cde(n.*acde(W0, k), k1);
% % semilogx(W, abs(Phi), '-r');
% H  = 1./(1+Phi.^2);
% H2 = 1./(1+Phi2.^2);
% % plot(W, abs((Phi)), '-r');
% semilogx(W, 10.*log10(abs(H)), '-r');
% hold on;
% semilogx(W, 10.*log10(abs(H2)), '--b');
% hold off;
% grid on;
% min(Phi);
% max(Phi);
% ylim([-100,0]);
% y = cde(5.*acde(W, k),k1);
% plot(W, abs(y), '-r');
% grid on;

% loglog(W,1./(1+(cde(5.*acde(W,0.8),0.01)).^2))
% a=[];
% for ii=2:20
%     n = ii;
%     a(ii-1)=2/(n.*(n+1));
% end
% semilogy(2:20, a);grid on;

% num1=[4.1931];
% den1=[1 4.2605 6.9293 4.1931];
% % num1=[2.7722];
% % den1=[1 3.4177 4.8669 2.7722];
% sys1 = tf(num1, den1);
% step(sys1, 3:0.01:10)
% fType = 'Gaussian';
% fType = 'Bessel';
% fType = 'Butterworth';
% TeeEn = 0;
% n     = 20;
% Rs    = 50;
% Rl    = 50;
% fp    = 1/2/pi;
% fs    = 10*fp;
% Ap    = 10*log10(2);
% As    = 60;
% bw    = 1;
% fShape = 'LPF';
% f0 = 1e-3;
% f1 = 10;
% N  = 100;
% [IdealFreq, IdealMag, IdealPhase, P, Z, f_min] = funSimFilterIdeal(fType, TeeEn, n, Rs, Rl, fp, fs, Ap, As, bw, fShape, f0, f1, N);
% Pr = P(real(P)<0).*2.*pi;
% den1 = abs(poly(Pr));
% num1 = [den1(end)];
% sys1 = tf(num1, den1);
% [yout,x,t] = step(sys1, 0:0.01:20);
% plot(x, yout, '-g');
% % semilogx(IdealFreq, IdealMag, '-r');
% grid on;
% fprintf('%0.3f%%\n', (max(yout)-1)*100);

% Rs = 2.27/3.27/(1+2.27/3.27);
% RL = 1;
% % R1 = 0.5;
% % Rs = 0.3;
% syms L1 C2 Z1 Z2 s
% R1 = (sqrt(5)-1)/2*RL
% Rs = R1*RL/(R1+RL)
% Z1 = RL*(1/(s*C2))/(RL+(1/(s*C2)));
% Z2 = Z1+s*L1;
% Z3 = Z2*R1/(R1+Z2);
% Z4 = Z3+Rs;
% 
% f = Z2/Z1*Z4/Z3;
% f_s = collect(f, s);
% coeff_f = coeffs(f_s, s)
% % coeff_target = 4/1.488*[1.488,0.976,0.488];
% % RLP = RL*R1/(RL+R1);
% coeff_target = fliplr([1,1.222,1.746])./1.746.*(2);
% % coeff_target = [1.220, 0.853, 0.699];
% eqns = arrayfun(@(a, b) a == b, coeff_f(2:end), coeff_target(2:end));
% sol = solve(eqns, L1, C2)
% vpa(sol.C2,4)
% vpa(sol.L1,4)
% vpa(sol.R1,4)
% R1 = linspace(0.01, 1.5, 100);
% y_c2 = (0.0005727.*(611.0.*R1 - 873.0.*(-1.312e-6.*(4.997e+5.*R1 - 7.466e+5).*(R1 + 2.0)).^(1/2) + 1222.0))./R1;
% y_l1 = (2.8.*(R1 + 2.0))./(R1 + 4.0) - (0.002291.*(611.0.*R1 - 873.0.*(-1.312e-6.*(4.997e+5.*R1 - 7.466e+5).*(R1 + 2.0)).^(1/2) + 1222.0))./(R1 + 4.0);
% y_c2 = (0.0005727.*(1222.0.*R1 - 873.0.*(-5.248e-6.*(4.997e+5.*R1 - 3.733e+5).*(R1 + 1.0)).^(1/2) + 1222.0))./R1;
% y_l1 = (1.4.*(R1 + 1.0))./(R1 + 2.0) - (0.0005727.*(1222.0.*R1 - 873.0.*(-5.248e-6.*(4.997e+5.*R1 - 3.733e+5).*(R1 + 1.0)).^(1/2) + 1222.0))./(R1 + 2.0);
% plot(R1, real(y_c2), '-*r');
% hold on;
% plot(R1, imag(y_c2), '-b');
% plot(R1, real(y_l1), '--*g');
% plot(R1, imag(y_l1), '--g');
% hold off;
% grid on;




% Rs = 1;
% syms L1 C2 L3 C4 Z1 Z2 Z3 Z4 s Rl
% 
% Z1 = Rl*(1/(s*C4))/(Rl+(1/(s*C4)));
% Z2 = Z1+s*L3;
% Z3 = Z2*(1/(s*C2))/(Z2+(1/(s*C2)));
% Z4 = Z3+Rs+s*L1;
% 
% f = Z2*Z4/(Z1*Z3);
% f_s = collect(f, s);
% coeff_f = coeffs(f_s, s);
% coeff_target = [1.25, 2.438, 3.001, 1.839, 0.925];
% eqns = arrayfun(@(a, b) a == b, coeff_f, coeff_target);
% sol = solve(eqns)
% plot(vpa(sol.C4), '*r');grid on; 
% ylim([-1,1]);
% xlim([-1,3]);
% axis equal;

% n=2;
% Rs = 1;
% Rl = 1;
% fp = 1;
% fs = 1;
% Ap = 10*log10(2);
% As = 10;
% N  = 100;
% x = linspace(1.02, 1.05, N);
% m = [];
% for ii=1:N
%     [cellValueNetlist, km, Rs] = funSynthesisLinearAmpFilter(n, Rs, Rl, fp, fs, Ap, As, x(ii));
%     m(ii) = km(1)*km(2);
% %     ii
% end
% plot(x, m, '-r', 'linewidth', 2);
% grid on;
% % 1.234*0.5739



