%%201713709057 Haluk MALGIR
function tahmin=tahminim(yol)
H=load('Egitilmis');
ADS2= audioDatastore(yol,'IncludeSubfolders',1);
ADS2.Labels = extractBetween(ADS2.Files,fullfile(yol,filesep),filesep);
Fs=H.Fs;

frameDuration = 200e-3;
overlapDuration = 40e-3;
frameLength = floor(H.Fs*frameDuration); 
overlapLength = round(H.Fs*overlapDuration);

[XTest1,YTest1]=preprocessAudioData(ADS2,frameLength,overlapLength,H.Fs);
YPred1 = classify(H.net,XTest1);
a = countcats(YPred1);
b = a(1);
for i = 1:5
    if a(i) >= b
        b = a(i);
        c = i;
    end
end

if c==5
  tahmin=('Konuşan Haluk');
else
   tahmin=('Konuşan Haluk değil');
end


function [X,Y] = preprocessAudioData(ADS,SL,OL,Fs)

    if ~isempty(ver('parallel'))
        pool = gcp;
        numPar = numpartitions(ADS,pool);
    else
        numPar = 1;
    end
    
    parfor ii = 1:numPar
    
        X = zeros(1,SL,1,0);
        Y = zeros(0);
        subADS = partition(ADS,numPar,ii);
        
        while hasdata(subADS)
            [audioIn,dsInfo] = read(subADS);
            audioIn=[audioIn(:,1)];
          
            
            speechIdx = detectSpeech(audioIn,Fs);
            numChunks = size(speechIdx,1);
            audioData = zeros(1,SL,1,0);      
            
            for chunk = 1:numChunks
                % Isaretlenen sesler temizleniyor
                audio_chunk = audioIn(speechIdx(chunk,1):speechIdx(chunk,2));
                audio_chunk = buffer(audio_chunk,SL,OL);
                q = size(audio_chunk,2);
                
                % Ses 200 ms'lik parçalara ayriliyor
                audio_chunk = reshape(audio_chunk,1,SL,1,q);
                
                % Mevcut sesle birleştir
                audioData = cat(4,audioData,audio_chunk);
            end
            
            audioLabel = str2double(dsInfo.Label{1});
            
            % Matrisi çoğaltarak eğitim ve test için etiketler oluşturun
            audioLabelsTrain = repmat(audioLabel,1,size(audioData,4));
            
            % Mevcut konuşmacı için veri isaretleniyor
            X = cat(4,X,audioData);
            Y = cat(2,Y,audioLabelsTrain);
        end
            
        XC{ii} = X;
        YC{ii} = Y;
    end  
    X = cat(4,XC{:});
    Y = cat(2,YC{:});
    Y = categorical(Y);    
end
end