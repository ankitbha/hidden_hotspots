function varargout = SemiVariog(varargin)
% SEMIVARIOG MATLAB code for SemiVariog.fig
%      SEMIVARIOG, by itself, creates a new SEMIVARIOG or raises the existing
%      singleton*.
%
%      H = SEMIVARIOG returns the handle to a new SEMIVARIOG or the handle to
%      the existing singleton*.
%
%      SEMIVARIOG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEMIVARIOG.M with the given input arguments.
%
%      SEMIVARIOG('Property','Value',...) creates a new SEMIVARIOG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SemiVariog_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SemiVariog_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SemiVariog

% Last Modified by GUIDE v2.5 23-Apr-2017 12:48:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SemiVariog_OpeningFcn, ...
                   'gui_OutputFcn',  @SemiVariog_OutputFcn, ...
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


% --- Executes just before SemiVariog is made visible.
function SemiVariog_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SemiVariog (see VARARGIN)

% Choose default command line output for SemiVariog
handles.output = hObject;
if ~isempty(varargin)
    handles.Src_data = varargin{1};
end
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SemiVariog wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SemiVariog_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure

if ~isfield(handles,'VariogFitPar_N')
    VarigPar.VariogFitPar_N = [];
else
    VarigPar.VariogFitPar_N = handles.VariogFitPar_N;
end
if ~isfield(handles,'VariogFitPar_E')
    VarigPar.VariogFitPar_E = [];
else
    VarigPar.VariogFitPar_E = handles.VariogFitPar_E;
end
if ~isfield(handles,'VariogFitPar_U')
    VarigPar.VariogFitPar_U = [];
else
    VarigPar.VariogFitPar_U = handles.VariogFitPar_U;
end

varargout{1} = VarigPar;
delete(handles.figure1);

% --- Executes on button press in CLOSE.
function CLOSE_Callback(hObject, eventdata, handles)
% hObject    handle to CLOSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

uiresume(handles.figure1);

function range_Callback(hObject, eventdata, handles)
% hObject    handle to range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of range as text
%        str2double(get(hObject,'String')) returns contents of range as a double


% --- Executes during object creation, after setting all properties.
function range_CreateFcn(hObject, eventdata, handles)
% hObject    handle to range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in FitExpVariogValue.
function FitExpVariogValue_Callback(hObject, eventdata, handles)
% hObject    handle to FitExpVariogValue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

direct_flag_N = get(handles.radio_N,'Value');
direct_flag_E = get(handles.radio_E,'Value');
direct_flag_U = get(handles.radio_U,'Value');
if direct_flag_N == 1
    S = get(handles.radio_N,'UserData');
    if isempty(S)
        errorDlg('You should calucate N direction Exp-Variogram Value firstly!');
        return;
    end
elseif direct_flag_E == 1
    S = get(handles.radio_E,'UserData');
    if isempty(S)
        errorDlg('You should calucate E direction Exp-Variogram Value firstly!');
        return;
    end
else
    S = get(handles.radio_U,'UserData');
    if isempty(S)
        errorDlg('You should calucate E direction Exp-Variogram Value firstly!');
        return;
    end
end

model_flag_spherical = get(handles.radio_spherical,'Value');
model_flag_gaussian = get(handles.radio_gaussian,'Value');
if model_flag_spherical == 1
    model_str = 'spherical';
elseif model_flag_gaussian == 1
    model_str = 'gaussian';
else
    model_str = 'exponential';
end

range = str2num(get(handles.range,'String'));
sill = str2num(get(handles.sill,'String'));
nugget = str2num(get(handles.nugget,'String'));

weight_flag_none = get(handles.radio_none,'Value');
weight_flag_cressie85 = get(handles.radio_cressie85,'Value');
if weight_flag_none == 1
    weight_str = 'none';
elseif weight_flag_cressie85 == 1
    weight_str = 'cressie85';
else
    weight_str = 'mcbratney86';
end

cla(handles.VarigPlot,'reset')
axes(handles.VarigPlot)
plot(S.distance,S.val,'^')
hold on
[a,c,n,SFit] = variogramfit(S.distance,S.val,range,sill,S.num,...
    'nugget',nugget,'weightfun',weight_str,'model',model_str);
SFit.trendfun = S.trendfun;
if direct_flag_N == 1
    handles.VariogFitPar_N = SFit;
elseif direct_flag_E == 1
    handles.VariogFitPar_E = SFit;
else
    handles.VariogFitPar_U = SFit;
end
guidata(hObject, handles);

function sill_Callback(hObject, eventdata, handles)
% hObject    handle to sill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sill as text
%        str2double(get(hObject,'String')) returns contents of sill as a double


% --- Executes during object creation, after setting all properties.
function sill_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nugget_Callback(hObject, eventdata, handles)
% hObject    handle to nugget (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nugget as text
%        str2double(get(hObject,'String')) returns contents of nugget as a double


% --- Executes during object creation, after setting all properties.
function nugget_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nugget (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nrbins_Callback(hObject, eventdata, handles)
% hObject    handle to nrbins (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nrbins as text
%        str2double(get(hObject,'String')) returns contents of nrbins as a double


% --- Executes during object creation, after setting all properties.
function nrbins_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nrbins (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxdist_Callback(hObject, eventdata, handles)
% hObject    handle to maxdist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxdist as text
%        str2double(get(hObject,'String')) returns contents of maxdist as a double


% --- Executes during object creation, after setting all properties.
function maxdist_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxdist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in PlotExpVariogValue.
function PlotExpVariogValue_Callback(hObject, eventdata, handles)
% hObject    handle to PlotExpVariogValue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.Src_data)
    errorDlg('Please Input data firstly!');
    return;
end
Src_data = handles.Src_data;
direct_flag_N = get(handles.radio_N,'Value');
direct_flag_E = get(handles.radio_E,'Value');
direct_flag_U = get(handles.radio_U,'Value');
x = Src_data.x; y = Src_data.y;
if direct_flag_N == 1 && ~isempty(Src_data.dn)
    data = Src_data.dn;
elseif direct_flag_E == 1 && ~isempty(Src_data.de)
    data = Src_data.de;
elseif direct_flag_U == 1 && ~isempty(Src_data.du)
    data = Src_data.du;
else
    errorDlg('No Data in this direction!');
    return;
end

[nrbins,nrbins_flag] = str2num(get(handles.nrbins,'String'));
if nrbins~=fix(nrbins)
    errorDlg('number bins the distance should be an integer!');
    return;
end

[maxdist,maxdist_flag] = str2num(get(handles.maxdist,'String'));
trend_flag_C = get(handles.Radio_constant,'Value');
trend_flag_L = get(handles.Radio_linear,'Value');
if trend_flag_C == 1
    fun_handle = @trendpoly0;
elseif trend_flag_L == 1
    fun_handle = @trendpoly1;
else
    fun_handle = @trendpoly3;
end

S = variog([x y],data,fun_handle,nrbins,maxdist);
S.trendfun = fun_handle;
S.nrbins = nrbins;
S.maxdist = maxdist;
if direct_flag_N == 1
    set(handles.radio_N,'UserData',S);
elseif direct_flag_E == 1
    set(handles.radio_E,'UserData',S);
else
    set(handles.radio_U,'UserData',S);
end
cla(handles.VarigPlot,'reset')
axes(handles.VarigPlot)
plot(S.distance,S.val,'^')
xlabel('lag distance h')
ylabel('\gamma(h)')


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
% delete(hObject);
uiresume(handles.figure1);


% --- Executes when selected object is changed in uipanel_Direction.
function uipanel_Direction_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uipanel_Direction 
% eventdata  structure with the following fields (see UIBUTTONGROUP)
%	EventName: string 'SelectionChanged' (read only)
%	OldValue: handle of the previously selected object or empty if none was selected
%	NewValue: handle of the currently selected object
% handles    structure with handles and user data (see GUIDATA)

direct_flag_N = get(handles.radio_N,'Value');
direct_flag_E = get(handles.radio_E,'Value');

if direct_flag_N == 1
    S = get(handles.radio_N,'UserData');
    if isempty(S)
        cla(handles.VarigPlot,'reset');
        set(handles.Radio_constant,'Value',1);
        set(handles.nrbins,'String','');
        set(handles.maxdist,'String','');
        return;
    else
        if isequal(S.trendfun,@trendpoly0)
            set(handles.Radio_constant,'Value',1);
        elseif isequal(S.trendfun,@trendpoly1)
            set(handles.Radio_linear,'Value',1);
        else
            set(handles.Radio_Cubic,'Value',1);
        end
        set(handles.nrbins,'String',num2str(S.nrbins));
        set(handles.maxdist,'String',num2str(S.maxdist));
        
        cla(handles.VarigPlot,'reset')
        axes(handles.VarigPlot)
        plot(S.distance,S.val,'^')
        xlabel('lag distance h')
        ylabel('\gamma(h)')
    end
elseif direct_flag_E == 1
    S = get(handles.radio_E,'UserData');
    if isempty(S)
        cla(handles.VarigPlot,'reset');
        set(handles.Radio_constant,'Value',1);
        set(handles.nrbins,'String','');
        set(handles.maxdist,'String','');
        return;
    else
        if isequal(S.trendfun,@trendpoly0)
            set(handles.Radio_constant,'Value',1);
        elseif isequal(S.trendfun,@trendpoly1)
            set(handles.Radio_linear,'Value',1);
        else
            set(handles.Radio_Cubic,'Value',1);
        end
        set(handles.nrbins,'String',num2str(S.nrbins));
        set(handles.maxdist,'String',num2str(S.maxdist));
        
        cla(handles.VarigPlot,'reset')
        axes(handles.VarigPlot)
        plot(S.distance,S.val,'^')
        xlabel('lag distance h')
        ylabel('\gamma(h)')
    end
else
    S = get(handles.radio_U,'UserData');
    if isempty(S)
        cla(handles.VarigPlot,'reset');
        set(handles.Radio_constant,'Value',1);
        set(handles.nrbins,'String','');
        set(handles.maxdist,'String','');
        return;
    else
        if isequal(S.trendfun,@trendpoly0)
            set(handles.Radio_constant,'Value',1);
        elseif isequal(S.trendfun,@trendpoly1)
            set(handles.Radio_linear,'Value',1);
        else
            set(handles.Radio_Cubic,'Value',1);
        end
        set(handles.nrbins,'String',num2str(S.nrbins));
        set(handles.maxdist,'String',num2str(S.maxdist));
        
        cla(handles.VarigPlot,'reset')
        axes(handles.VarigPlot)
        plot(S.distance,S.val,'^')
        xlabel('lag distance h')
        ylabel('\gamma(h)')
    end
end