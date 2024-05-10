
clc;
clear all;
close all;

%==========================================================================
% Uploading Different accelerometers data and ploting the uploaded data
%==========================================================================

load MobileAccData30orientationRawSample01
RawAccData= MobileAccData30orientationRawSample01;

% Computing the mean of Raw Acc Data
MeanRawAccData(1,1:3) = mean(RawAccData(1:335,:));
MeanRawAccData(2,1:3) = mean(RawAccData(377:641,:));
MeanRawAccData(3,1:3) = mean(RawAccData(824:1005,:));
MeanRawAccData(4,1:3) = mean(RawAccData(1162:1508,:));
MeanRawAccData(5,1:3) = mean(RawAccData(1608:1855,:));

MeanRawAccData(6,1:3) = mean(RawAccData(1922:2251,:));
MeanRawAccData(7,1:3) = mean(RawAccData(2318:2608,:));
MeanRawAccData(8,1:3) = mean(RawAccData(2701:2909,:));
MeanRawAccData(9,1:3) = mean(RawAccData(2977:3099,:));
MeanRawAccData(10,1:3) = mean(RawAccData(3213:3323,:));

MeanRawAccData(11,1:3) = mean(RawAccData(3455:3570,:));
MeanRawAccData(12,1:3) = mean(RawAccData(3651:3775,:));
MeanRawAccData(13,1:3) = mean(RawAccData(3869:3959,:));
MeanRawAccData(14,1:3) = mean(RawAccData(4040:4178,:));
MeanRawAccData(15,1:3) = mean(RawAccData(4221:4348,:));

MeanRawAccData(16,1:3) = mean(RawAccData(4435:4519,:));
MeanRawAccData(17,1:3) = mean(RawAccData(4611:4756,:));
MeanRawAccData(18,1:3) = mean(RawAccData(4855:4960,:));
MeanRawAccData(19,1:3) = mean(RawAccData(5070:5116,:));
MeanRawAccData(20,1:3) = mean(RawAccData(5243:5345,:));

MeanRawAccData(21,1:3) = mean(RawAccData(5425:5548,:));
MeanRawAccData(22,1:3) = mean(RawAccData(5613:5736,:));
MeanRawAccData(23,1:3) = mean(RawAccData(5825:5975,:));
MeanRawAccData(24,1:3) = mean(RawAccData(6082:6194,:));
MeanRawAccData(25,1:3) = mean(RawAccData(6264:6377,:));

MeanRawAccData(26,1:3) = mean(RawAccData(6412:6580,:));
MeanRawAccData(27,1:3) = mean(RawAccData(6663:6781,:));
MeanRawAccData(28,1:3) = mean(RawAccData(6843:6955,:));
MeanRawAccData(29,1:3) = mean(RawAccData(7014:7151,:));
MeanRawAccData(30,1:3) = mean(RawAccData(7216:7264,:));


% V_orig = MeanRawAccData(1:20,1:3);
% 
% % Computing local variation in gravity
% e=0.0818191908425 ;%eccentricity of the ellipsoid, e,
% Lat=31.2010; %latitude
% lon=121.432; % longitude
% g_theo=9.8321849378;
% g_local= 9.7803253359*(1 + 0.001931853*sin(Lat)^2)/sqrt(1-e^2*sin(Lat)^2);
% g_local= convacc(g_local,'m/s^2','G''s');


% g=[-1.224641e-01  -2.986263e-03  -1.233589e-01 ]'
%  g_2= convacc(g,'m/s^2','G''s');
% g3=[-40e-3; g_2; 40e-3;]
% figure
% plot(g3,'bs--')

% load AutocalibrationOfMEMSAccelerometers;
% V_orig=AutocalibrationOfMEMSAccelerometers;
% g_local=1;

load CalibrationOfLowCostTriaxialInertialSensors30
V_orig=CalibrationOfLowCostTriaxialInertialSensors30;
g_local=1;
% (1:3,1:18)
% CalibrationOfLowCostTriaxialInertialSensors30=CalibrationOfLowCostTriaxialInertialSensors50(1:3,15:45)';
% save CalibrationOfLowCostTriaxialInertialSensors30.mat
V_inaccur=V_orig;
 V_inaccur(1,3)=V_orig(1,3);
% V_inaccur(3,3)=V_orig(3,3)-1;
V=V_inaccur;


figure
plot(V_orig);
grid on;
grid minor;

figure
plot(V);
grid on;
grid minor;

% give ERROR contain less than 10 orientations
[r, c] = size(V);
if r < 9
    disp('Need atleast 9 Measurements for the calibration procedure!')
    return
end
if c ~= 3
    disp('Not enough columns in the data')
    return
end


%==========================================================================
% Initial Guess values of M and B.
%==========================================================================

% Initial Guess values of M and B.
% Mxx0 = 1;
% Mxy0 = 0.02;
% Mxz0 = 0.02;
% Myy0 = 1;
% Myz0 = 0.02;
% Mzz0 = 1;
% Bx0 = 0.02;
% By0 = 0.02;
% Bz0 = 0.02;

% Initial Guess values of M and B.
Mxx0 = 5;
Mxy0 = 0.5;
Mxz0 = 0.5;
Myy0 = 5;
Myz0 = 0.5;
Mzz0 = 5;
Bx0 = 0.5;
By0 = 0.5;
Bz0 = 0.5;

% Initial Guess values of M and B.
v = [Mxx0, Mxy0, Mxz0, Myy0, Myz0, Mzz0, Bx0, By0, Bz0]';
XaO_GL=v;
XaO=v;



%==========================================================================
% defing the variable for Calibaration parmeter and raw data
% error model
%==========================================================================
% defining the variable for Calibaration parmeter and raw data
sa = sym('sa',[3 3]);
sa(2,1)=sa(1,2);
sa(3,1)=sa(1,3);
sa(3,2)=sa(2,3);

ba = sym('ba',[3 1]);                      % Define the bias vector for gyro as bias bi => R3
va = sym('Va',[3 1]);
XaVar=[sa(1,1:3).'; sa(2,2); sa(2,3); sa(3,3); ba;];  % For finding the Jaccobian
XaVarSubs=[sa(1,1:3).'; sa(2,2); sa(2,3); sa(3,3); ba; va;];  % For subsituting the values in function

% defining the error Model, cost Function and finding Jaccobian
model=sa*(va- ba);
funResid=norm(model)-g_local;
funCost=(norm(model)-g_local)^2;
JacAcc(1,1:9)= jacobian(funResid, XaVar);


%==========================================================================
% finding the error of orignal and noised data and plot them
%==========================================================================
R_orig_GL = zeros(length(V), 1);
Rb_GL= zeros(length(V), 1);

for i=1:length(V)
    XaO_GL(10:12)=V(i,1:3);
    Rb_GL(i,1)=double(subs(funResid, XaVarSubs, XaO_GL));
end


for i=1:length(V_orig)
    XaO_GL(10:12)=V_orig(i,1:3);
    R_orig_GL(i,1)=double(subs(funResid, XaVarSubs, XaO_GL));
end

Rb_GL=Rb_GL./norm(Rb_GL);
R_orig_GL=R_orig_GL./norm(R_orig_GL);


figure
plot(R_orig_GL,'b')
grid on
grid minor
hold on
plot(Rb_GL,'ro');
title('error comparasion');


%==========================================================================
% finding the Weight of orignal and noised data and plot them
%==========================================================================

nn=length(V);


CC_GL=diag(Rb_GL*Rb_GL');
W2=(diag(1./CC_GL));
W2=diag(W2)./norm(diag(W2));
W=diag(W2);

CC_orig_GL=diag(R_orig_GL*R_orig_GL');
Worig2=(diag(1./CC_orig_GL));
Worig=diag(Worig2)./norm(diag(Worig2));


% RR_GL=(Rb_GL-(mean(Rb_GL)*ones(length(Rb_GL),1))).^2;
% for i=1:length(RR_GL)
%     CC_GL(i,1)=sum(RR_GL(1:i))/i+1;
% end
% ccc_GL=diag(RR_GL);
% W=inv(ccc_GL);
% 
% figure
% plot(RR_GL)
% 
% figure
% plot(diag(W))


% RR_orig_GL=(R_orig_GL-mean(R_orig_GL)*ones(length(R_orig_GL),1)).^2;
% for i=1:length(RR_orig_GL)
%     CC_orig_GL(i,1)=sum(RR_orig_GL(1:i))/i+1;
% end
% CCC_orig_GL=diag(CC_orig_GL);
% Worig=inv(CCC_orig_GL);

figure
plot((Worig),'ro')
grid on
grid minor
hold on
plot(diag(W),'b');
title('weight')


%==========================================================================
%  Configurable variables % defifning the parameters for itrative loop
%==========================================================================
lambda = 1;      % Damping Gain - Start with 1
kl = 0.01;       % Damping paremeter - has to be less than 1. Changing this will affect rate of convergence ; Recommend to use k1 between 0.01 - 0.05
tol = 1e-9;      % Convergence criterion threshold
itr = 50;        % No. Of iterations. If your solutions don't converge then try increasing this. Typically it should converge within 20 iterations
format short
% format longE
tXvar=1:length(V);


R_GL = zeros(length(V), 1);
Rcost_GL = zeros(length(V), 1);
J_GL = zeros(length(V), 2);

XaO_GL=v;
Xest_GL=XaO_GL;
Rold_GL=100000;   % Better to leave this No. big.

%==========================================================================
% Generlized least square itrative loop
%==========================================================================
for k=0:itr;
    
    for i=1:length(V)
        XaO_GL(10:12)=V(i,1:3);
        R_GL(i,1)=double(subs(funResid, XaVarSubs, XaO_GL));
        Rcost_GL(i,1)=double(subs(funCost, XaVarSubs, XaO_GL));
        J_GL(i,1:9) = double(subs(JacAcc, XaVarSubs, XaO_GL));
    end
    
    Rnew_GL=sum(W*Rcost_GL)
    SSR_GL(k+1)=R_GL'*R_GL;
    gradient_GL=J_GL'*W*R_GL;
    grad_GL= norm(gradient_GL)
    
    %----------------------------------------------------------------
    A=J_GL'*W*J_GL;
    B=J_GL'*W*R_GL;
    DeltaX=inv(A)*B;
    Xest_GL=Xest_GL-DeltaX;
    %-----------------------------------------------------------------
    
    % Iterations are stopped when the following convergence criteria is
    % satisfied
    if  (k>1)
        sprintf('%d', abs(max(2*(Xest_GL-XestOld_GL)/(Xest_GL+XestOld_GL))));
%         if (abs(max(2*(Xest_GL-XestOld_GL)/(Xest_GL+XestOld_GL))) <= tol)
               if (norm(gradient_GL) <= tol)
            disp('Convergence achieved');
            break;
               end
    end
    
    XaO_GL=Xest_GL;
    XestOld_GL=Xest_GL;
    Rold_GL=Rnew_GL;
    disp(k);
end


%==========================================================================
% Displaying Outputs
%==========================================================================
figure
plot(SSR_GL,'r')

S_current_GL = [Xest_GL(1) Xest_GL(2) Xest_GL(3); Xest_GL(2) Xest_GL(4) Xest_GL(5); Xest_GL(3) Xest_GL(5) Xest_GL(6)];
B_current_GL = [Xest_GL(7);Xest_GL(8);Xest_GL(9)];
disp('Bias By Generalize Least Square ')
disp(B_current_GL)
% fprintf('.16g',B_GLeastSquare)
disp('ScaleFactor By Generalize Least Square')
disp(S_current_GL)
disp('cost Function of Generalize Least Square')
disp(Rnew_GL)


%==========================================================================
% comparing the result
%==========================================================================
BGL=[ 8.532274200335077e-04
    -5.184294747401816e-03
     1.353838368977337e-02];
SGL=   [  1.007513695523271e+00    -7.730988628065297e-03     2.655083930615642e-02
    -7.730988628065297e-03     9.969318790230274e-01     1.760436497070423e-03
     2.655083930615642e-02     1.760436497070423e-03     9.933545093801416e-01];

deltaB_GL=abs(BGL-B_current_GL);
deltaS_GL=abs(SGL-S_current_GL);

disp('Change in Bias of GLS in percent');
disp(deltaB_GL);
disp('Change in Scale Factor of GLS in percent');
disp(deltaS_GL);
%==========================================================================






R_LM = zeros(length(V), 1);
Rcost_LM = zeros(length(V), 1);
J_LM = zeros(length(V), 2);

XaO_LM=v;
Xest_LM=XaO_LM;
Rold_LM=100000;   % Better to leave this No. big.

Ndata=length(V);
Nparams=length(XaO_LM); % a and b are the parameters to be estimated
%==========================================================================
% LM itrative loop
%==========================================================================
for k=0:itr % iterate
    
    for i=1:length(V)
        XaO_LM(10:12)=V(i,1:3);
        R_LM(i,1)=double(subs(funResid, XaVarSubs, XaO_LM));
        Rcost_LM(i,1)=double(subs(funCost, XaVarSubs, XaO_LM));
        J_LM(i,1:9) = double(subs(JacAcc, XaVarSubs, XaO_LM));
    end
    
    Rnew_LM=sum(Rcost_LM)
    R_LM'*R_LM
    SSR_LM(k+1)=R_LM'*R_LM;
    gradient_LM=J_LM'*R_LM;
    grad_LM= norm(gradient_LM)
    
    %---------------------------
    H_LM=J_LM'*J_LM;
    H_lm=H_LM+(lambda*eye(Nparams,Nparams));
    Xest_LM=Xest_LM-inv(H_lm)*(J_LM'*R_LM(:));
    %------------------------------
    
    
    
    % This is to make sure that the error is decereasing with every
    if (Rnew_LM <= Rold_LM)
        lambda=lambda/10;
    else % otherwise increases the value of the damping factor
        lambda=lambda*10;
    end
    
    
    % Iterations are stopped when the following convergence criteria is
    % satisfied
    if  (k>1)
        sprintf('%d', abs(max(2*(Xest_LM-vold_LM)/(Xest_LM+vold_LM))));
        if (abs(max(2*(Xest_LM-vold_LM)/(Xest_LM+vold_LM))) <= tol)
            %         if (norm(gradient) <= tol)
            disp('Convergence achieved');
            break;
        end
    end
    
    XaO_LM = Xest_LM;
    vold_LM = Xest_LM;
    Rold_LM = Rnew_LM;
    disp(k);
    
end


%==========================================================================
% Displaying Outputs
%==========================================================================
figure
plot(SSR_LM,'r')

% Save Outputs
S_current_LM = [Xest_LM(1) Xest_LM(2) Xest_LM(3); Xest_LM(2) Xest_LM(4) Xest_LM(5); Xest_LM(3) Xest_LM(5) Xest_LM(6)];
B_current_LM = [Xest_LM(7);Xest_LM(8);Xest_LM(9)];
disp('Bias By Levenberg-Marquardt  ')
disp(B_current_LM)
disp('ScaleFactor By Levenberg-Marquardt ')
disp(S_current_LM)
disp('cost Function of Levenberg-Marquardt ')
disp(Rnew_LM)


%==========================================================================
% comparing the result
%==========================================================================
B_LM=[ 8.532240217593794e-04
    -5.181945094812548e-03
     1.355429934560954e-02];
S_LM=   [   1.007516546325410e+00    -7.742453453974131e-03     2.654482949930700e-02
    -7.742453453974131e-03     9.969083367564701e-01     1.774606878513853e-03
     2.654482949930700e-02     1.774606878513853e-03     9.933325355310924e-01 ];

deltaB_LM=abs(B_LM - B_current_LM);
deltaS_LM=abs(S_LM - S_current_LM);

disp('Change in Bias of LM in percent');
disp(deltaB_LM);
disp('Change in Scale Factor of LM in percent');
disp(deltaS_LM);
%==========================================================================








R_GN = zeros(length(V), 1);
Rcost_GN = zeros(length(V), 1);
J_GN = zeros(length(V), 2);

XaO_GN=v;
Xest_GN=XaO_GN;
Rold_GN=100000;   % Better to leave this No. big.
lambda = 1;      % Damping Gain - Start with 1


%==========================================================================
% GN itrative loop
%==========================================================================
for k=0:itr % iterate
    
    for i=1:length(V)
        XaO_GN(10:12)=V(i,1:3);
        R_GN(i,1)=double(subs(funResid, XaVarSubs, XaO_GN));
        Rcost_GN(i,1)=double(subs(funCost, XaVarSubs, XaO_GN));
        J_GN(i,1:9) = double(subs(JacAcc, XaVarSubs, XaO_GN));
    end
    
    Rnew_GN=sum(Rcost_GN)
    SSR_GN(k+1)=Rnew_GN;
    gradient_GN=J_GN'*R_GN;
    grad_GN= norm(gradient_GN)

    %--------------------------------------------------------------------
    H_GN = inv(J_GN'*J_GN); % Hessian matrix
    D_GN = (J_GN'*R_GN)';
    Xest_GN = Xest_GN - lambda*(D_GN*H_GN)';
    %   v=v-lambda*inv(J'*J)*J'*R;
    %-------------------------------------------------------------
    
    % This is to make sure that the error is decereasing with every
    % iteration
    if (Rnew_GN <= Rold_GN)
        lambda = lambda-kl*lambda;
    else
        lambda = kl*lambda;
    end
    
    % Iterations are stopped when the following convergence criteria is
    % satisfied
    if  (k>1)
        sprintf('%d', abs(max(2*(Xest_GN-XestOld_GN)/(Xest_GN+XestOld_GN))));
        if (abs(max(2*(Xest_GN-XestOld_GN)/(Xest_GN+XestOld_GN))) <= tol)
            disp('Convergence achieved');
            break;
        end
    end
    
    XaO_GN = Xest_GN;
    XestOld_GN = Xest_GN;
    Rold_GN = Rnew_GN;
    disp(k);
end

%==========================================================================
% Displaying Outputs
%==========================================================================
figure
plot(SSR_GN,'r')

S_current_GN = [Xest_GN(1) Xest_GN(2) Xest_GN(3); Xest_GN(2) Xest_GN(4) Xest_GN(5); Xest_GN(3) Xest_GN(5) Xest_GN(6)];
B_current_GN = [Xest_GN(7);Xest_GN(8);Xest_GN(9)];
disp('Bias By GaussNewton ')
disp(B_current_GN)
disp('ScaleFactor By GaussNewton')
disp(S_current_GN)
disp('cost Function of GaussNewton')
disp(Rnew_GN)


%==========================================================================
% comparing the result
%==========================================================================    
    B_GN=[   8.532239735284780e-04
    -5.181945154549919e-03
     1.355429927809855e-02];
    S_GN=[   1.007516546403092e+00    -7.742453459550752e-03     2.654482949894881e-02
    -7.742453459550752e-03     9.969083368329824e-01     1.774606875538570e-03
     2.654482949894881e-02     1.774606875538570e-03     9.933325356079964e-01];
    disp('CalibrationOfLowCostTriaxialInertialSensors')

deltaB_GN=abs(B_GN-B_current_GN);
deltaS_GN=abs(S_GN-S_current_GN);

disp('Change in Bias og GN');
disp(deltaB_GN);
disp('Change in Scale Factor of GN');
disp(deltaS_GN);
%============================================================================



%===================================================================================================================
% Using Non linear least square Method By Matlab comand "lsqnonlin" for estimating the calibration parameters
%====================================================================================================================

% Fg = Funct_Gauss(XaVar,g_local,V);
options = optimset('Algorithm',{'levenberg-marquardt',0.1});
lb=-inf;
ub=inf;
[phiLSQ,resnorm,residual]=lsqnonlin(@(phi) Funct_Gauss(phi,g_local,V), XaO,lb,ub, options);
saf=phiLSQ(1:6);
B_lsqnonlin =phiLSQ(7:9);   % Define the bias vector for gyro as bias bi => R3
S_lsqnonlin=[saf(1) saf(2) saf(3);
    saf(2) saf(4) saf(5);
    saf(3) saf(5) saf(6);];
disp('Bias By lsqnonlin ')
disp(B_lsqnonlin)
disp('ScaleFactor By lsqnonlin ')
disp(S_lsqnonlin)
disp('cost Function of lsqnonlin')
disp(resnorm)


[phiLSQJac,resnormJac,residualJac]=lsqnonlin(@(phi) Funct_Gauss(phi,g_local,V), XaO);




%---------------------------------------------------------------------------------------------------------------------
% Using Levenberg-Marquardt algorithm Method By Matlab comand "LMFnlsq" for estimating the calibration parameters
Options = LMFnlsq; %   for setting default values for Options,
XXo=XaO;
[Xf, Ssq, CNT, Res, XY] = LMFnlsq(@(phi) Funct_Gauss(phi,g_local,V),XXo,Options);

sa_LMFnlsq=Xf(1:6);
B_LMFnlsq =Xf(7:9);   % Define the bias vector for gyro as bias bi => R3
S_LMFnlsq=[sa_LMFnlsq(1) sa_LMFnlsq(2) sa_LMFnlsq(3);
    sa_LMFnlsq(2) sa_LMFnlsq(4) sa_LMFnlsq(5);
    sa_LMFnlsq(3) sa_LMFnlsq(5) sa_LMFnlsq(6);];

disp('Bias By LMFnlsq ')
disp(B_LMFnlsq)
disp('ScaleFactor By LMFnlsq')
disp(S_LMFnlsq)
disp('cost Function of LMFnlsq')
disp(Ssq)


%==================================================================================

SSR_LM
SSR_GN
SSR_GL

figure('position',[500 110 470 300])
subplot(3,1,1)
plot(SSR_LM,'k-o','LineWidth',1.5)
ylabel('RMSE')
legend('LMA')

subplot(3,1,2)
plot(SSR_GN,'k-o','LineWidth',1.5)
ylabel('RMSE')
legend('GNA')

subplot(3,1,3)
plot(SSR_GL,'k-o','LineWidth',1.5)
xlabel('Number of itrations')
ylabel('RMSE')
legend('GNLSA')
set(gcf,'color','white')



figure('position',[500 110 470 300])
plot(SSR_LM,'r','LineWidth',1)
hold on
plot(SSR_GN,'g','LineWidth',1)
hold on
plot(SSR_GL,'b:','LineWidth',3)
xlabel('Number of itrations')
ylabel('RMSE')
legend('Levenberg-Marquardt','Gauss-Newton','Generalized least square')
set(gcf,'color','white')

% ============================================================================
% Verifivaion of Result

load MobileDataZaxis
MobileData = MobileDataZaxis;
figure
plot(MobileData)

set(gcf,'color','white')
% computing the Calibrated X
calibratedData_GLeastSquare=zeros(length(MobileData),3);
for i=1:length(MobileData)
    calibratedData_GLeastSquare(i,1:3)=S_current_GN*((MobileData(i,1:3)'- B_current_GN));
end
MeasuredGravity_GLeastSquare=[mean(calibratedData_GLeastSquare(:,1)) mean(calibratedData_GLeastSquare(:,2)) mean(calibratedData_GLeastSquare(:,3))];
AccCompToGravitynorm_GLeastSquare=norm(MeasuredGravity_GLeastSquare)-g_local;


Ref_g=zeros(length(MobileData),3);
for i=1:length(MobileData)
    Ref_g(i,1:3)=[0 0 g_local];
end

Traw=1:length(MobileData);
Tref=1:length(Ref_g);


%-------------------------------


figure('position',[500 110 470 300])
subplot(3, 2 ,1), 
plot(Traw,MobileData(:,1),'k');
title('Raw(uncalibrated) Measurement')
ylabel('A_x(m/s^2)')
axis([0,length(Traw),-1,1])
grid on

subplot(3, 2 ,2)
 plot(Traw,calibratedData_GLeastSquare(:,1),'k');
title('Calibrated Measurement')
% ylabel('A_x(m/s^2)')
axis([0,length(Traw),-1,1])
grid on


subplot(3, 2 ,3)
plot(Traw,MobileData(:,2),'k');
ylabel('A_y(m/s^2)')
axis([0,length(Traw),-1,1])
grid on

subplot(3, 2 ,4)
plot(Traw,calibratedData_GLeastSquare(:,2),'k');
% ylabel('A_y(m/s^2)')
axis([0,length(Traw),-1,1])
grid on


subplot(3, 2 ,5)
plot(Traw,MobileData(:,3),'k');
ylabel('A_z(m/s^2)')
xlabel('Numbers of samples')
axis([0,length(Traw),9,10.5])
grid on

subplot(3, 2 ,6)
plot(Traw,calibratedData_GLeastSquare(:,3),'k');
xlabel('Numbers of samples')
axis([0,length(Traw),9,10.5])
grid on




%----------------------------------





figure
subplot(3, 1 ,1)
plot(Traw,MobileData(:,1),'b');
% hold on
% plot(Tref,Ref_g(:,1),'r');
hold on
plot(Traw,calibratedData_GLeastSquare(:,1),'g');
legend('Raw output','Calibrated output')
xlabel('Numbers of samples')
ylabel('X axis Output (m/s^2)')
set(gcf,'color','white')

subplot(3, 1 ,2)
plot(Traw,MobileData(:,2),'b');
hold on
% plot(Tref,Ref_g(:,2),'r');
% hold on
plot(Traw,calibratedData_GLeastSquare(:,2),'g');
legend('Raw output','Calibrated output')
xlabel('Numbers of samples')
ylabel('Y axis Output (m/s^2)')
set(gcf,'color','white')


subplot(3, 1 ,3)
plot(Traw,MobileData(:,3),'b');
hold on
% plot(Tref,Ref_g(:,3),'r');
% hold on
plot(Traw,calibratedData_GLeastSquare(:,3),'g');
legend('Raw output','Calibrated output')
xlabel('Numbers of samples')
ylabel('Z axis Output (m/s^2)')
set(gcf,'color','white')




figure
plot(Traw,calibratedData_GLeastSquare(:,1),'b');
hold on
plot(Traw,calibratedData_GLeastSquare(:,2),'r');
hold on
plot(Traw,calibratedData_GLeastSquare(:,3),'g');
title('calibrated Data of Accelerometer')
set(gcf,'color','white')

figure
plot(Traw,MobileData(:,1),'b');
hold on
plot(Traw,MobileData(:,2),'r');
hold on
plot(Traw,MobileData(:,3),'g');
legend('fx','fy','fz');
title('Raw Data of Accelerometer')
set(gcf,'color','white')


figure
plot(Tref,Ref_g(:,1),'b');
hold on
plot(Tref,Ref_g(:,2),'r');
hold on
plot(Tref,Ref_g(:,3),'g');
legend('fx','fy','fz');
title('Reference of Accelerometer')
set(gcf,'color','white')

