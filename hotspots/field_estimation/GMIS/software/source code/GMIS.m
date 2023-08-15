function varargout = GMIS(varargin)
% GMIS MATLAB code for GMIS.fig
%      GMIS, by itself, creates a new GMIS or raises the existing
%      singleton*.
%
%      H = GMIS returns the handle to a new GMIS or the handle to
%      the existing singleton*.
%
%      GMIS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GMIS.M with the given input arguments.
%
%      GMIS('Property','Value',...) creates a new GMIS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GMIS_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GMIS_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GMIS

% Last Modified by GUIDE v2.5 23-Apr-2017 11:43:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GMIS_OpeningFcn, ...
                   'gui_OutputFcn',  @GMIS_OutputFcn, ...
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


% --- Executes just before GMIS is made visible.
function GMIS_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GMIS (see VARARGIN)

% Choose default command line output for GMIS
handles.output = hObject;
set(handles.List_Seris,'string','...');
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GMIS wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GMIS_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in List_Seris.
function List_Seris_Callback(hObject, eventdata, handles)
% hObject    handle to List_Seris (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns List_Seris contents as cell array
%        contents{get(hObject,'Value')} returns selected item from List_Seris

currentSel = get(handles.List_Seris,'Value');
if isempty(currentSel)
    return;
end
drawSerisFig(currentSel,handles)

% --- Executes during object creation, after setting all properties.
function List_Seris_CreateFcn(hObject, eventdata, handles)
% hObject    handle to List_Seris (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function FILE_Callback(hObject, eventdata, handles)
% hObject    handle to FILE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function SEMIVARIOGFIT_Callback(hObject, eventdata, handles)
% hObject    handle to SEMIVARIOGFIT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Src_data = get(handles.List_Seris,'UserData');
VarigPar = SemiVariog(Src_data);
set(handles.SEMIVARIOGFIT,'UserData',VarigPar);

% --------------------------------------------------------------------
function KKFINTERP_Callback(hObject, eventdata, handles)
% hObject    handle to KKFINTERP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

VarigPar = get(handles.SEMIVARIOGFIT,'UserData');
Src_data = get(handles.List_Seris,'UserData');
InterpResult = EMInterp(Src_data,VarigPar);
if isempty(InterpResult)
    InterpResult.interpN = [];
    InterpResult.interpE = [];
    InterpResult.interpU = [];
end
set(handles.OPEN,'UserData',InterpResult.interpN);
set(handles.SAVE,'UserData',InterpResult.interpE);
set(handles.CLOSE,'UserData',InterpResult.interpU);
currentSel = get(handles.List_Seris,'Value');
drawSerisFig(currentSel,handles);

% --------------------------------------------------------------------
function ABOUT_Callback(hObject, eventdata, handles)
% hObject    handle to ABOUT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

About();

% --------------------------------------------------------------------
function OPEN_Callback(hObject, eventdata, handles)
% hObject    handle to OPEN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Open a mat File
[MatNameWithTag, MatDirectory] = uigetfile({'*.mat','Mat Files'},...
            'Select a mat file');
if MatNameWithTag == 0
    errorDlg('You should select an Mat File');
    return;
end

%load data
Src_data = load(strcat(MatDirectory,MatNameWithTag));
filedNames = fieldnames(Src_data);
if length(filedNames) ~= 1
    errorDlg('There should be only one struct in Mat File!');
    return;
end
expr = strcat('Src_data = Src_data.',filedNames,';');
eval(expr{:});

%check Input Filed Names
miss_InterpFileds = {'day' 
    'dn'
    'de'
    'du'
    'x'
    'y'
    'site'};
for i=1:length(miss_InterpFileds)
    if ~isfield(Src_data, miss_InterpFileds{i})
        errorDlg(strcat(miss_InterpFileds{ i },' Field Not Found!'));
        return;
    end
end

%check value day
if isempty(Src_data.day)
    errorDlg('There is no data in Mat File!');
    return;
elseif ~isvector(Src_data.day)
    errorDlg('day filed value must be a vector1');
    return;
end

%check value site
if isempty(Src_data.site)
    errorDlg('There is no site in Mat File!');
    return;
elseif ~ischar(Src_data.site)
    errorDlg('Site Name must be char value!');
    return;
end

lenDay = length(Src_data.day);
numSite = size(Src_data.site,1);

%check N/E/U data
if isempty(Src_data.dn)
%     warning('There is no data in N direction!');
elseif ~ismatrix(Src_data.dn)
    errorDlg('dn filed value must be a two dimensional matrix!');
    return;
elseif size(Src_data.dn,1) ~= lenDay || ...
        size(Src_data.dn,2) ~= numSite
    errorDlg({'dn filed value size can not compare';...
        ' length of day or site!'});
    return;
end

if isempty(Src_data.de)
%     warning('There is no data in E direction!');
elseif ~ismatrix(Src_data.de)
    errorDlg('de filed value must be a two dimensional matrix!');
    return;
elseif size(Src_data.de,1) ~= lenDay || ...
        size(Src_data.de,2) ~= numSite
    errorDlg({'de filed value size can not compare ';...
        ' length of day or site!'});
    return;
end

if isempty(Src_data.du)
%     warning('There is no data in U direction!');
elseif ~ismatrix(Src_data.du)
    errorDlg('de filed value must be a two dimensional matrix!');
    return;
elseif size(Src_data.du,1) ~= lenDay || ...
        size(Src_data.du,2) ~= numSite
    errorDlg({'de filed value size can not compare';...
        ' length of day or site!'});
    return;
end

%
set(handles.OPEN,'UserData',[]);
set(handles.SAVE,'UserData',[]);
set(handles.CLOSE,'UserData',[]);
set(handles.List_Seris,'string','...');
set(handles.List_Seris,'Value',1);
set(handles.SEMIVARIOGFIT,'UserData',[]);

%show source data in List box
set(handles.List_Seris,'string',Src_data.site);
currentSel = get(handles.List_Seris,'Value');
set(handles.List_Seris,'UserData',Src_data);
drawSerisFig(currentSel,handles);

% --------------------------------------------------------------------
function SAVE_Callback(hObject, eventdata, handles)
% hObject    handle to SAVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[MatFileName,MatPathName]=uiputfile({
    '*.mat','Mat Files(*.mat)';'*.*','All Files(*.*)'},'choose a File');
if MatFileName == 0
    errordlg('You should save any Mat File');
    return;
end
Interp_data = get(handles.List_Seris,'UserData');
Interp_data.interp_dn = get(handles.OPEN,'UserData');
Interp_data.interp_de = get(handles.SAVE,'UserData');
Interp_data.interp_du = get(handles.CLOSE,'UserData');
Interp_data.VarigPar = get(handles.SEMIVARIOGFIT,'UserData');
save(strcat(MatPathName,MatFileName),'Interp_data');

% --------------------------------------------------------------------
function CLOSE_Callback(hObject, eventdata, handles)
% hObject    handle to CLOSE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);


% --- Executes during object creation, after setting all properties.
function axes_N_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes_N (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes_N
box on
axes(hObject);
set(hObject,'Xticklabel',[])

% --- Executes during object creation, after setting all properties.
function axes_E_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes_E (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes_E
box on
set(hObject,'Xticklabel',[])

% --- Executes during object creation, after setting all properties.
function axes_U_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes_U (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes_U
box on


function drawSerisFig(Sel,handles)
if strcmp(get(handles.List_Seris,'string'), '...');
    return;
end

cla(handles.axes_N,'reset')
set(handles.axes_N,'box', 'on','Xticklabel',[])

cla(handles.axes_E,'reset')
set(handles.axes_E,'box', 'on','Xticklabel',[])

cla(handles.axes_U,'reset')
set(handles.axes_U,'box', 'on')

Src_data = get(handles.List_Seris,'UserData');
if isempty(Src_data)
    return;
end
axes(handles.axes_N);
if ~isempty(Src_data.dn)
    plot(Src_data.day,Src_data.dn(:,Sel),'k');
end
Interp_dn = get(handles.OPEN,'UserData');
if ~isempty(Interp_dn)
    hold on
    plot(Src_data.day,Interp_dn(:,Sel),'r');
    h = legend('Original data','Filter and Interpolated data',...
        'Orientation','horizontal');
    set(h,'Position', [112.8333   41.6026   67.5333    1.8462])
    hold off
end


datetick(handles.axes_N,'x','yyyy/mm');
set(handles.axes_N, 'XTickLabel',[],'box','on')
ylabel(handles.axes_N,'N','FontSize',10);
xlim(handles.axes_N,[Src_data.day(1),Src_data.day(end)])

axes(handles.axes_E);
if ~isempty(Src_data.de)
    plot(Src_data.day,Src_data.de(:,Sel),'k');
end
Interp_de = get(handles.SAVE,'UserData');
if ~isempty(Interp_de)
    hold on
    plot(Src_data.day,Interp_de(:,Sel),'r');
    if ~ishandle(h)
        h = legend('Original data','Filter and Interpolated data',...
        'Orientation','horizontal');
        set(h,'Position', [112.8333   41.6026   67.5333    1.8462])
    end
    hold off
end
datetick(handles.axes_E,'x','yyyy/mm');
set(handles.axes_E,'XTickLabel',[],'box','on')
ylabel(handles.axes_E,'E','FontSize',10);
xlim(handles.axes_E,[Src_data.day(1),Src_data.day(end)])

axes(handles.axes_U);
if ~isempty(Src_data.du)
plot(Src_data.day,Src_data.du(:,Sel),'k');
end
Interp_du = get(handles.CLOSE,'UserData');
if ~isempty(Interp_du)
    hold on
    plot(Src_data.day,Interp_du(:,Sel),'r');
    if ~ishandle(h)
        h = legend('Original data','Filter and Interpolated data',...
        'Orientation','horizontal');
        set(h,'Position', [112.8333   41.6026   67.5333    1.8462])
    end
    hold off
end
datetick(handles.axes_U,'x','yyyy/mm');
ylabel(handles.axes_U,'U','FontSize',10);
xlabel(handles.axes_U,'Year/month','FontSize',10)
xlim(handles.axes_U,[Src_data.day(1),Src_data.day(end)])

function errorDlg(str)

h = errordlg(str);
set(h,'windowStyle','modal')
g = handle(h);
g.javaFrame.fHG1Client.getWindow.setAlwaysOnTop(true);
