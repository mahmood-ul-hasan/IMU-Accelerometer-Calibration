function [F,J]= Funct_Gauss (phi,g_local,V)


saf=phi(1:6);
baf=phi(7:9);   % Define the bias vector for gyro as bias bi => R3
sa_matf=[saf(1) saf(2) saf(3); 
        saf(2) saf(4) saf(5);
        saf(3) saf(5) saf(6);];
% Scale factor matrix is symmetric
% sa(2,1)=sa(1,2);
% sa(3,1)=sa(1,3);
% sa(3,2)=sa(2,3);

for i=1:length(V)
    h_xf(3*i-2:3*i,1)=sa_matf*(V(i,1:3)'-baf);
     F(i,1)= norm(h_xf(3*i-2:3*i,1))-g_local;

end 

if nargout > 1
%  compting the jaccobian
J= jacobian(h_x, phi);
end

end 
%     

