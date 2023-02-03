function [Sigma, D, nu, Delta] = dim_red4(Sigma, D, nu, Delta,cut_tol)
    % Reduces the dimension of csn according to the correlations of the conditions

    n  = length(Sigma);
    P  = [Sigma, Sigma*D'; D*Sigma, Delta+D*Sigma*D'];
    P  = 0.5*(P + P');
    n2 = length(P);

    try
        stdnrd = diag(1./sqrt(diag(P)));
        Pcorr  = abs(stdnrd*P*stdnrd);
    catch
        Sigma = 0;
        D     = 0;
        nu    = 0;
        Delta = 0;
        return
    end

    Pcorr = Pcorr-diag(repelem(Inf, n2));
    logi2 = max(Pcorr(n+1:end, 1:n), [], 2);
    
%     % Notify if there are more than ... dimensions left uncut
%     if sum(logi2 > 0.1) > 25
%         disp('----');
%         disp(max(logi2)); disp(sum(logi2 > 0.1));
%         disp('----');
%     end

    logi2 = (logi2 < cut_tol);

%     if sum(~logi2) > 25
%         disp(sum(~logi2))
%     end
    
    %Cut unnecessary dimensions
    D(logi2, :)     = [];
    nu(logi2)       = [];
    Delta(logi2, :) = [];
    Delta(:, logi2) = [];

end