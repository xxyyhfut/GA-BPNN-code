
%% The code is a prediction code based on a genetic algorithm neural network
% Clear environment variables
clc
clear

load state4  % Load dataset
%% Training data collation

Theoretical_angle=state_4_JOINT;
input_Px1=Theoretical_angle(:,7); %Reset grating 1 position (mm)
input_Px2=Theoretical_angle(:,8); %Reset grating 2 position (mm)

output_joint_actual=Theoretical_angle(:,11); % joint angle (deg)

% NN_input_7D=[input_Px1,input_Px2,input_Pv1,input_Pv2,input_Mx,input_MV,input_joint_desired];
NN_input_4D=[input_Px1,input_Px2];%Array collection of input data

NN_output_3D=[output_joint_actual];%Array collection of output data

nn=length(input_Px1);
n_train=round(nn*0.9); % 90% of the data was randomly selected as training input and the remaining 10% as predicted data
n_test=nn-n_train;

step=0.01; % Sampling period 10ms
t=step:step:(nn*step);
%% Network structure establishment
%gensim(net,-1)
input_SEANN=NN_input_4D; % Neural network training input
output_SEANN=NN_output_3D; % Neural network training output

%Number of nodes
inputnum=2;
hiddennum=10;
outputnum=1;


%Total number of nodes
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

% Training data and prediction data % need to be reprocessed, how to input and output multidimensional data
k=rand(1,nn); % 90% of the data is randomly selected as the training input and the remaining 10% as the prediction data
[m,n]=sort(k);
input_train=input_SEANN(n(1:n_train),:)';
input_test=input_SEANN(n((n_train+1):nn),:)';
output_train=output_SEANN(n(1:n_train),:)';
output_test=output_SEANN(n((n_train+1):nn),:)';

% Selected sample input/output data normalization
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

% Build network
net=newff(inputn,outputn,hiddennum);

%% Genetic algorithm parameter initialization
maxgen=20;                         % Evolutionary algebra, i.e. the number of iterations
sizepop=10;                        % Population size
pcross=[0.2];                       % Cross probability selection, between 0 and 1
pmutation=[0.1];                    % Variation probability choice, between 0 and 1

lenchrom=ones(1,numsum);        
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    % Data range

%------------------------------------------------------Population initialization--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  % Defines population information as a structure
avgfitness=[];                     % Average fitness of each generation population
bestfitness=[];                     % Optimum fitness of each generation population
bestchrom=[];                       % The best fit chromosome
% Initialize the population
for i=1:sizepop
    % Randomly generate a population
    individuals.chrom(i,:)=Code(lenchrom,bound);    % encoding (binary and grey encoding result is a real number, float encoding result is a real vector)
    x=individuals.chrom(i,:);
    % Calculation fitness
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   % Fitness of chromosomes
end
FitRecord=[];
% Finding the best chromosomes
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  % The best chromosome
avgfitness=sum(individuals.fitness)/sizepop; % Average fitness of chromosomes
% Record the best and average fitness in each generation of evolution
trace=[avgfitness bestfitness]; 
 
%% Iteratively solve for the best initial thresholds and weights
% Initiation of evolution
for i=1:maxgen
    i
    % Select
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    % intersect
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % Mutation
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % Computational fitness 
    for j=1:sizepop
        x=individuals.chrom(j,:); %decode
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
  % Find the chromosomes of minimum and maximum fitness and their position in the population
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    % Replacing the best chromosome from the last evolution
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    
    trace=[trace;avgfitness bestfitness]; %Record the best and average fitness in each generation of evolution
    FitRecord=[FitRecord;individuals.fitness];
end

%% Analysis of genetic algorithm results
figure(1)
[r c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['fitness curve  ' 'terminating algebra£½' num2str(maxgen)]);
xlabel('evolutionary algebra');ylabel('adaptation');
legend('Average adaptation','optimal adaptation');
disp('adaptation              variant');

%% Assigns the optimal initial threshold weight to the network prediction
% % BP network optimized by genetic algorithm is used for value prediction
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP network training
%network evolution parameter
net.trainParam.epochs=100; % Number of training sessions is 100
net.trainParam.lr=0.005; % Learning rate is 0.1
%network training
[net,per2]=train(net,inputn,outputn);

%% BP network prediction
%data normalization
inputn_test=mapminmax('apply',input_test,inputps); % Normalization of forecast data
an=sim(net,inputn_test); %Network Predictive Output
test_simu=mapminmax('reverse',an,outputps);%Network output inverse normalization
error=test_simu-output_test;%prediction error

% Store training models and results
GEN_BPNN_2022_6_6_JOINT_23_net=net;
GEN_BPNN_2022_6_6_JOINT_23_inputps=inputps;
GEN_BPNN_2022_6_6_JOINT_23_outputps=outputps;
GEN_BPNN_2022_6_6_JOINT_23_input_train=input_train;
GEN_BPNN_2022_6_6_JOINT_23_input_test=input_test;
GEN_BPNN_2022_6_6_JOINT_23_output_train=output_train;
GEN_BPNN_2022_6_6_JOINT_23_output_test=output_test;
GEN_BPNN_2022_6_6_JOINT_23_error=error;

save GEN_BPNN_2022_6_6_JOINT_23_net=net
save GEN_BPNN_2022_6_6_JOINT_23_inputps=inputps
save GEN_BPNN_2022_6_6_JOINT_23_outputps=outputps
save GEN_BPNN_2022_6_6_JOINT_23_input_train=input_train
save GEN_BPNN_2022_6_6_JOINT_23_input_test=input_test
save GEN_BPNN_2022_6_6_JOINT_23_output_train=output_train
save GEN_BPNN_2022_6_6_JOINT_23_output_test=output_test
save GEN_BPNN_2022_6_6_JOINT_23_error=error
%% Result analysis 10% sample data test analysis
x_time=round(nn*0.1)+0.1; % x axis length
figure(2)
subplot(3,1,1)
plot(test_simu','o')
hold on
plot(output_test','*')
legend('joint-actual','joint-estimated','DCT-actual','DCT-estimated','BCT-actual','BCT-estimated');
title('Predicted output and target values using GEN-BPNN');
ylabel('Output/deg+N','fontsize',12);
xlim([0 x_time])
% xlabel('Random sequence of test samples','fontsize',12);

subplot(3,1,2)
plot(error','o')
% title('Absolute error of predicted outputs');
legend('joint angle-ae','DCT-ae','BCT-ae');
ylabel('Absolute errors','fontsize',12);
xlim([0 x_time])
% xlabel('Random sequence of test samples','fontsize',12);
rms_test=rms(error')

subplot(3,1,3)
error_relt=(output_test-test_simu)./test_simu*100;
plot(error_relt','p')
% title('Relative errors');
legend('joint angle-re');
ylabel('Relative errors/%','fontsize',12);
xlim([0 x_time])
xlabel('Random sequence of test samples','fontsize',12);


view(net)

