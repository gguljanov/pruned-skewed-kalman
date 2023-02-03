function log_res = logcdf_ME(Zj, Corr_mat)

    % cdf_ME  : Evaluate approximate log(CDF) according to Mendell Elston
    % Zj      : A column vector of points where CDF is evaluated, of size (len_cdf)
    % Corr_mat: Correlation matrix of size (len_cdf) x (len_cdf)
    %
    % Written by dbauer (Dietmar Bauer, Bielefeld University), 22.9.2015 
    % Modified by gguljanov (Gaygysyz Guljanov, Muenster University), 27.9.2022
    
    cutoff = 6;
    
    Zj(Zj>cutoff)  = cutoff;
    Zj(Zj<-cutoff) = -cutoff;
    
    len_cdf = length(Zj); % dimension of CDF.
    
    cdf_val = phid(Zj(1));
    pdf_val = phip(Zj(1));
    
    log_res = log(cdf_val); % perform all calcs in logs.
    
    for jj = 1:(len_cdf-1)

        ajjm1 = pdf_val / cdf_val;
        
        % Update Zj and Rij
        tZ = Zj + ajjm1 * Corr_mat(:, 1);    % update Zj
        
        R_jj = Corr_mat(:, 1) * Corr_mat(1, :);
        tRij = Corr_mat - R_jj * ( ajjm1 + Zj(1) ) * ajjm1;    % update Rij
        
        % Convert Rij (i.e. Covariance matrix) to Correlation matrix
        cov2corr = sqrt( diag(tRij) );
        
        Zj = tZ ./ cov2corr;
        Corr_mat = tRij ./ (cov2corr * cov2corr');
        
        % Cutoff those dimensions if they are too low to be evaluated
        cutoff = 38;
    
        Zj(Zj>cutoff)  = cutoff;
        Zj(Zj<-cutoff) = -cutoff;

        % Evaluate jj's probability
        cdf_val = phid( Zj(2) );
        pdf_val = phip( Zj(2) );

        % Delete unnecessary parts of updated Zj and Corr_mat
        Zj(1) = [];
        Corr_mat(1, :) = [];
        Corr_mat(:, 1) = [];
        
        % Overall probability
        log_res = log_res + log(cdf_val);
        
    end

end % End cdfmvna_ME


%%%%%%%%%%%%%%
% Normal PDF
%%%%%%%%%%%%%%
function p = phip(z)

    % Written by dbauer (Dietmar Bauer, Bielefeld University)

    p = exp(-z.^2/2)/sqrt(2*pi); % Normal pdf

end


%%%%%%%%%%%%%%
% Normal CDF
%%%%%%%%%%%%%%
function p = phid(z)
    
    %
    % taken from Alan Gentz procedures.
    %
    % Written by dbauer (Dietmar Bauer, Bielefeld University)

    p = erfc( -z/sqrt(2) )/2; % Normal cdf

end



