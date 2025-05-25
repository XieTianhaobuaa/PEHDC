clear;clc;
%Perform K pulsar period estimations
K=100;
for u=1:K
    %Load the trained networks CNN1 and CNN2
    load('CNN1.mat')
    load('CNN2.mat')
    %Set the simulation duration
    T=1000;
    %Simulate and generate pulsar photon data
    DATA = Data_sp(T);
    %Calculate the size of the pulsar sequence data
    [m,n] = size(DATA);
    TOA1=[];
    %Save the information of the photon arrival time
    TOA1(:,1)=DATA(:,1)-DATA(1,1)*ones(m,1);
    %Save the energy information carried by photons
    TOA1(:,2)=DATA(:,2);
    %Set the optimized energy threshold
    T1=500;
    T2=5000;
    %Filter energy
    h=1;
    for f=1:length(TOA1(:,1))
        if TOA1(f,2)<T2&&TOA1(f,2)>T1
            TOA(h,1)=TOA1(f,1);
            TOA(h,2)=TOA1(f,2);
            h=h+1;
        end
    end
    %Calculate the size of the pulsar sequence data after optimizing the energy
    [m,n] = size(TOA);
    %Set the number of bins
    bin_num = BIN;
    chi_range = 1;
    %Lower bound of the search cycle
    eP_lowerbound = P1;
    %Upper bound of the search cycle
    eP_upperbound = P2;
    %Search step size
    eP_step = 0.0000000001;
    for P = eP_lowerbound:eP_step:eP_upperbound
        Tb = P/bin_num;
        %Perform the photon epoch folding operation
        profile = Epoch_folding(bin_num,P,Tb,m,TOA);
        %Standardized photon folding profile
        profile= 255*(profile - min(profile)) / (max(profile) - min(profile));
        %Import the Hibert template
        load('muban.mat')
        %Perform Hilbert encoding
        for o=1:16
            for p=1:16
                for q=1:256
                    if muban(o,p)==q
                        profile_new(o,p)=profile(q,1);
                    end
                end
            end
        end
        %Two-dimensional image preprocessing
        profile_new=imresize(profile_new, [224 224]);
        profile_new=im2double(profile_new);
        data_ready=profile_new;
        %Convert the image data from grayscale to RGB
        profile_rgb = cat(3, data_ready, data_ready,data_ready);
        %Store the image data in an array
        test_data(:, :, :, 1) = profile_rgb;
        %Extract the label of 1 in CNN1
        predictedLabels = classify(CNN1, test_data);
        chi2(chi_range,1) = double(predictedLabels)-1;
        %Extract the output feature vectors in CNN2
        a=activations(CNN2, test_data, 'fc');
        chi2(chi_range,2) = double(a(:,:,1));
        chi2(chi_range,3) = double(a(:,:,2));
        load('S1.mat')
        load('S2.mat')
        %Calculate the minimum feature vector distance
        chi2(chi_range,4) = (chi2(chi_range,2)+S1)^2+(chi2(chi_range,3)+S2)^2;
        chi_range =  chi_range+1;
    end
    x=1;
    for i=1:length(chi2)
        if chi2(i,1)==1
            box(x,1)=i;
            box(x,2)=chi2(i,4);
            x=x+1;
        end
    end
    %Extract the index of the nearest distance
    [va,id]=min(box(:,2));
    P = (box(id,1)-1)*eP_step+eP_lowerbound; 
    PE_HDC(u,1)=abs(1e9*(P-0.0337))
end
