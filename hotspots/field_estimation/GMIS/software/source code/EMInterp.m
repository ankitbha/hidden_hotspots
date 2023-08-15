function varargout = EMInterp(varargin)
% EMINTERP MATLAB code for EMInterp.fig
%      EMINTERP, by itself, creates a new EMINTERP or raises the existing
%      singleton*.
%
%      H = EMINTERP returns the handle to a new EMINTERP or the handle to
%      the existing singleton*.
%
%      EMINTERP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EMINTERP.M with the given input arguments.
%
%      EMINTERP('Property','Value',...) creates a new EMINTERP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EMInterp_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EMInterp_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EMInterp

% Last Modified by GUIDE v2.5 23-Apr-2017 16:59:51

%% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EMInterp_OpeningFcn, ...
                   'gui_OutputFcn',  @EMInterp_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before EMInterp is made visible.
function EMInterp_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to EMInterp (see VARARGIN)

% Choose default command line output for EMInterp
handles.output = hObject;

handles.Src_data = varargin{1};
handles.VarigPar = varargin{2};
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes EMInterp wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = EMInterp_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


InterpResult = get(handles.Interp,'UserData');
% Get default command line output from handles structure
varargout{1} = InterpResult;
delete(handles.figure1);



function iters_Callback(hObject, eventdata, handles)
% hObject    handle to iters (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of iters as text
%        str2double(get(hObject,'String')) returns contents of iters as a double


% --- Executes during object creation, after setting all properties.
function iters_CreateFcn(hObject, eventdata, handles)
% hObject    handle to iters (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Interp.
function Interp_Callback(hObject, eventdata, handles)
% hObject    handle to Interp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Src_data = handles.Src_data;
VarigPar = handles.VarigPar;
if isempty(Src_data)
    errorDlg('Please Input data firstly!');
    return;
end

if isempty(VarigPar)
    VarigPar.VariogFitPar_N = [];
    VarigPar.VariogFitPar_E = [];
    VarigPar.VariogFitPar_U = [];
end

iters = str2num(get(handles.iters,'String'));
if iters~=fix(iters)
    errorDlg('number of EM iterations should be an integer!');
    return;
end

if isempty(iters)
    iters = 20;
end

ToUnitFlag_N = get(handles.check_N,'Value');
ToUnitFlag_E = get(handles.check_E,'Value');
ToUnitFlag_U = get(handles.check_U,'Value');

ToFiterFlag = get(handles.radio_Kalman_filter,'Value');
interp_n = EMEst(Src_data.x,Src_data.y,Src_data.dn,...
    VarigPar.VariogFitPar_N,ToUnitFlag_N,ToFiterFlag,iters,1);
interp_e = EMEst(Src_data.x,Src_data.y,Src_data.de,...
    VarigPar.VariogFitPar_E,ToUnitFlag_E,ToFiterFlag,iters,2);
interp_u = EMEst(Src_data.x,Src_data.y,Src_data.du,...
    VarigPar.VariogFitPar_U,ToUnitFlag_U,ToFiterFlag,iters,3);
Interp_data.interpN = interp_n;
Interp_data.interpE = interp_e;
Interp_data.interpU = interp_u;

set(handles.Interp,'UserData',Interp_data);
uiresume(handles.figure1);

function interp = EMEst(x,y,Z,VarigPar,ToUnitFlag,...
    ToFiterFlag,iters,direction)
%%
if isempty(Z)
    interp = [];
    return;
end

%% initial interpolate missing data
Ztmp = zeros(size(Z));
%interpolated
for i=1:size(Z,2)
    tmpZ = Z(:,i);
    day = (1:size(Z,1))';
    ij = isnan(tmpZ);
    tmpZ(ij)=[];
    day(ij) = [];
    Ztmp(:,i) = interp1(day,tmpZ,(1:size(Z,1))','linear');
end
%extrapolated
for i=1:size(Z,2)
    tmpZ = Ztmp(:,i);
    day = (1:size(Z,1))';
    ij = isnan(tmpZ);
    tmpZ(ij)=[];
    day(ij) = [];
    p = polyfit(day,tmpZ,1);
    day = (1:size(Z,1))';
    Ztmp(ij,i) = polyval(p,day(ij));
end


%% %%%%%%%%%%%%compute EM initial values%%%%%%%%%%%%%%%%%%
if isempty(VarigPar) || ToUnitFlag == 1
    %unit matrix instead spatial filed
    H = eye(size(Z,2));
    
    %observation noise covariance matrix
    sei2_error = 25;
    R = sei2_error*eye(size(H,1));
    
    %covariance matrix of initial state vector
    P0 = 0.5*R;
    
    %state transition matrix
    C1 = zeros(size(Z,2),size(Z,2));
    for i=1:(size(Z,1)-1)
        C1 = C1 + (Ztmp(i+1,:)'*Ztmp(i,:))/size(Z,1);
    end
    C0 = zeros(size(Z,2),size(Z,2));
    for i=1:size(Z,1)
        C0 = C0 + (Ztmp(i,:)'*Ztmp(i,:))/size(Z,1);
    end
    F = C1/C0;
    
    %system state noise covariance matrix
    Q = C0-C1/C0*C1';

else
    %spatial filed
    H = SpatialFiled(x,y,VarigPar,0.98);
    p = size(H,2);
    
    %observation noise covariance matrix
    R = VarigPar.nugget*eye(size(H,1));
    
    %covariance matrix of initial state vector
    [Qs,Rs]=qr(H,0);
    P0 = inv(Rs)*Qs'*(VarigPar.sill*eye(size(H,1)))*Qs*(inv(Rs))';
    
%     alpha_temp = zeros(size(Z,1),p);
%     for i=1:size(Z,1)
%         alpha_temp(i,:) = ((H'*H)\H'*Ztmp(i,:)')';
%     end
%     %state transition matrix
%     C1 = zeros(p,p);
%     for i=1:(size(Z,1)-1)
%         C1 = C1 + (alpha_temp(i+1,:)'*alpha_temp(i,:))/size(Z,1);
%     end
%     C0 = zeros(p,p);
%     for i=1:size(Z,1)
%         C0 = C0 + (alpha_temp(i,:)'*alpha_temp(i,:))/size(Z,1);
%     end
%     F = C1/C0;
%     
%     %system state noise covariance matrix
%     Q = C0-C1/C0*C1';
    
    %state transition matrix
    r = 0.8;
    F = r*eye(size(H,2));
    
    %system state noise covariance matrix
    Q = 0.9*(VarigPar.sill+VarigPar.nugget)*eye(p);
%     Q = 5*(VarigPar.sill+VarigPar.nugget)*eye(p);

end

alpha0 = zeros(size(H,2),1);

%% %%%%%%%%%%%%%%%%%%EM iteration%%%%%%%%%%%%%%%%%%
if ToFiterFlag==1
    interp = EMEst_filter(Z,H,...
        F,alpha0,P0,R,Q,iters,direction);
else
    interp = EMEst_smooth(Z,H,...
        F,alpha0,P0,R,Q,iters,direction);
end

% --- Executes on button press in check_N.
function check_N_Callback(hObject, eventdata, handles)
% hObject    handle to check_N (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_N


% --- Executes on button press in check_E.
function check_E_Callback(hObject, eventdata, handles)
% hObject    handle to check_E (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_E


% --- Executes on button press in check_U.
function check_U_Callback(hObject, eventdata, handles)
% hObject    handle to check_U (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_U

function errorDlg(str)

h = errordlg(str);
set(h,'windowStyle','modal')
g = handle(h);
g.javaFrame.fHG1Client.getWindow.setAlwaysOnTop(true);


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
uiresume(handles.figure1);
