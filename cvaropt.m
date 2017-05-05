%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% David Wang, Engineering Science 1T8+PEY
% MIE377 - Financial Optimization Models
% cvaropt.m | Lab 6 Soln 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear
clc

load('data502.mat');

rets=prices(2:end,:)./prices(1:end-1,:)-1; %Each row is a realization
mu=mean(rets);
Q=cov(rets);

alpha=0.95; %Tolerance

%% Mean Variance Opt
mvo=efV2(mu,Q);

%CVaR from the portfolios using return realizations
for i=1:50
    pret=rets*mvo.x(:,i);   %The return of the portfolio across the weekly samples
    var=prctile(pret,100-100*alpha); %5th percentile of returns is 95th percentile of losses
    z=(pret<var); %identify worst scenarios
    S2=sum(z); %number of scenarios used to compute cvar
    z=z.*pret;  %replace 1 with the return in that case
    mvo.cvar(i,1)=-sum(z)/S2;   %cvar is in loss, negative of return
end
clearvars i S2 z var pret
subplot(2,1,1)
plot(mvo.cvar,mvo.exp_ret) %this isnt smooth, and it shouldnt be since we optimized for minimum variance, not for CVaR
%efficient frontier should be smooth when optimized for CVaR below

%% CVaR optimization

S=210;
n=50;

%first 50 vars are x, next one is gamma, last 210 are z
lb=[zeros(n,1);-100;zeros(S,1)];

%add the expected return constraint, no restriction, find min risk
A=[-mu zeros(1,S+1)];
b=100;

%constrain weights to sum to 1
Aeq=[ones(1,n) zeros(1,S+1)];
beq=1;

%add the Z constraints
A=[A;-rets -ones(S,1) -eye(S)];
b=[b;zeros(S,1)];

%the objective
c=[zeros(1,n) 1 ones(1,S)/(S*(1-alpha))];

%solve, pass empty matrices for f, A, b, and ub
x=linprog(c,A,b,Aeq,beq,lb,[]);

%minimum risk portfolio
minret=mu*x(1:n);

%maximum return portfolio
x=linprog([-mu zeros(1,S+1)],A,b,Aeq,beq,lb,[]);

%maximum expected return
maxret=mu*x(1:n);

%for an efficient frontier of 50 points
stepsize=(maxret-minret)/49;

i=1;
for ret_targ=minret:stepsize:maxret
    b(1,1)=-ret_targ;
    cvar.x(:,i)=linprog(c,A,b,Aeq,beq,lb,[]);
    cvar.exp_ret(i,1)=(mu*cvar.x(1:n,i));
    cvar.risk(i,1)=((cvar.x(1:n,i)'*Q*cvar.x(1:n,i)))^.5;
    cvar.cv(i,1)=c*cvar.x(:,i);
    i=i+1;
end

hold on
plot(cvar.cv,cvar.exp_ret,'red');
title('Expected Return vs CVaR')
legend('MVO optimized','CVaR optimized')
subplot(2,1,2)
plot(mvo.risk,mvo.exp_ret)
hold on
plot(cvar.risk,cvar.exp_ret,'red');
title('Expected Return vs Standard Deviation')
legend('MVO optimized','CVaR optimized')
