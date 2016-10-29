function final_homo =  dogleg_func(homo,inlier)
    uk = 0.5;%constant defined for delta_gn calculation
    rk = 3;%constant defined
    term = 1;%the termination condition, pDL
    
    pk = [homo(1,1),homo(1,2),homo(1,3),homo(2,1),homo(2,2),homo(2,3),homo(3,1),homo(3,2),homo(3,3)]';
    
    
    while (term > 0)
        Jk = [];
        Ek = [];
        N = max(size(inlier));
        for i = 1:1:N %generate Jacobian and error 
            xk = inlier(i,1);%from image 1
            yk = inlier(i,2);
            xkm = inlier(i,3);%from image 2, matched
            ykm = inlier(i,4);
            xhat = pk(1)*xk + pk(2)*yk + pk(3);
            yhat = pk(4)*xk + pk(5)*yk + pk(6);
            what = pk(7)*xk + pk(8)*yk + pk(9);
            Jk = [Jk;xk/what, yk/what, 1/what,0,0,0, -xhat*xk/what^2, -xhat*yk/what^2, -xhat/what^2;0,0,0,xk/what, yk/what, 1/what,-yhat*xk/what^2, -yhat*yk/what^2, -yhat/what^2];
            Ek = [Ek,xkm - (xhat/what),ykm - (yhat/what)];
        end
        Ek = Ek';
        %compute delta_gd and delta_gn, norm(A) returns the 2-norm of matrix A
        d_gd =(norm(Jk'*Ek)/norm(Jk*Jk'*Ek))*Jk'*Ek;
        d_gn = (Jk'*Jk + uk*eye(9))\Jk'*Ek;

        %compute the update information
        if (norm(d_gn) < rk)
            update = d_gn;
        elseif ((norm(d_gd) < rk) && (rk < norm(d_gn))) %need to solve beta
            tmp1 =  d_gd'*d_gd - rk^2;
            tmp2 =  d_gd'*(d_gn-d_gd);
            tmp3 = (d_gn-d_gd)'*(d_gn-d_gd);
            beta = (((tmp2^2 - tmp1*tmp3)^0.5) - tmp2) / tmp3; 
            update = d_gd + beta*(d_gn - d_gd);
        else %otherwise
            update = rk*d_gd/norm(d_gd);
        end

        %Update information
        Cpk = Ek' * Ek;
        pk = pk + update;

        Jk_up = [];
        Ek_up = [];
        %calculate another step based on the update

        for i = 1:1:N %generate Jacobian and error 
            xk = inlier(i,1);%from image 1
            yk = inlier(i,2);
            xkm = inlier(i,3);%from image 2, matched
            ykm = inlier(i,4);
            xhat = pk(1)*xk + pk(2)*yk + pk(3);
            yhat = pk(4)*xk + pk(5)*yk + pk(6);
            what = pk(7)*xk + pk(8)*yk + pk(9);
            Jk_up = [Jk_up;xk/what, yk/what, 1/what,0,0,0, -xhat*xk/what^2, -xhat*yk/what^2, -xhat/what^2;0,0,0,xk/what, yk/what, 1/what,-yhat*xk/what^2, -yhat*yk/what^2, -yhat/what^2];
            Ek_up = [Ek_up,xkm - (xhat/what),ykm - (yhat/what)];
        end
        Ek_up = Ek_up';
        %Update information
        Cpk_up = Ek_up' * Ek_up;

        %Update terminate condition
        term =  (Cpk - Cpk_up)/(2*update'*Jk'*Ek - update'*Jk'*Jk*update);

       % Update rk
        if(term<0.25)
            rk = rk/4;
        elseif(term<=0.75)
            rk = rk;
        else
            rk = 2*rk;
        end
        
    end%end of while loop       
    final_homo = [pk(1),pk(2),pk(3);pk(4),pk(5),pk(6);pk(7),pk(8),pk(9)];%sudo solution
    
end