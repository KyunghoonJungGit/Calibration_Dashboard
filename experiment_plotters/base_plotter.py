"""
Base Plotter Module
모든 실험 플로터의 추상 기본 클래스를 정의
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np


class ExperimentPlotter(ABC):
    """모든 실험 플로터의 기본 클래스"""
    
    def __init__(self):
        """플로터 초기화"""
        # 공통 설정
        self.MAX_COLS = 2  # 최대 열 수
        self.VERTICAL_SPACING = 0.15  # 수직 간격
        self.HORIZONTAL_SPACING = 0.10  # 수평 간격
        self.SUBPLOT_HEIGHT = 300  # 각 서브플롯 기본 높이 (픽셀)
        
        # 색상 팔레트
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#adb5bd',
            'dark': '#343a40'
        }
        
        # 플롯 템플릿
        self.plot_template = "plotly_white"
    
    @property
    @abstractmethod
    def experiment_type(self) -> str:
        """이 플로터가 처리하는 실험 타입"""
        pass
    
    @abstractmethod
    def get_display_options(self) -> List[Any]:
        """실험별 디스플레이 옵션 컴포넌트 반환
        
        Returns
        -------
        List[dbc.Component]
            Dash 컴포넌트 리스트
        """
        pass
    
    @abstractmethod
    def create_plot(self, 
                   exp_data: Dict, 
                   selected_qubits: List[str], 
                   plot_options: Dict) -> go.Figure:
        """플롯 생성
        
        Parameters
        ----------
        exp_data : Dict
            실험 데이터 (ds_raw, ds_fit, qubit_info 등 포함)
        selected_qubits : List[str]
            선택된 큐빗들의 grid location
        plot_options : Dict
            플롯 옵션
            
        Returns
        -------
        go.Figure
            생성된 Plotly figure
        """
        pass
    
    @abstractmethod
    def get_default_options(self) -> Dict:
        """기본 플롯 옵션
        
        Returns
        -------
        Dict
            기본 옵션 값들
        """
        pass
    
    def get_subplot_layout(self, n_qubits: int, 
                          rows_multiplier: int = 1) -> Tuple[int, int, float]:
        """공통 서브플롯 레이아웃 계산
        
        Parameters
        ----------
        n_qubits : int
            큐빗 수
        rows_multiplier : int
            행 수 배수 (예: both 옵션일 때 2)
            
        Returns
        -------
        rows : int
            행 수
        cols : int
            열 수
        total_height : float
            전체 figure 높이
        """
        cols = min(self.MAX_COLS, n_qubits)
        rows = (n_qubits + cols - 1) // cols
        rows *= rows_multiplier
        
        total_height = rows * self.SUBPLOT_HEIGHT + 200  # 여백 포함
        
        return rows, cols, total_height
    
    def create_subplots_figure(self, rows: int, cols: int, 
                             subplot_titles: List[str],
                             total_height: float) -> go.Figure:
        """서브플롯 figure 생성
        
        Parameters
        ----------
        rows : int
            행 수
        cols : int  
            열 수
        subplot_titles : List[str]
            서브플롯 제목들
        total_height : float
            전체 높이
            
        Returns
        -------
        go.Figure
            생성된 figure
        """
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=self.VERTICAL_SPACING / rows if rows > 1 else 0,
            horizontal_spacing=self.HORIZONTAL_SPACING
        )
        
        # 기본 레이아웃 설정
        fig.update_layout(
            height=total_height,
            template=self.plot_template,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def update_subplot_titles_position(self, fig: go.Figure, titles: List[str]):
        """서브플롯 타이틀 위치 조정"""
        for annotation in fig['layout']['annotations']:
            if annotation['text'] in titles:
                annotation['y'] = annotation['y'] - 0.02  # 타이틀을 약간 아래로
    
    def apply_common_layout(self, fig: go.Figure, title: str):
        """공통 레이아웃 설정 적용
        
        Parameters
        ----------
        fig : go.Figure
            업데이트할 figure
        title : str
            플롯 제목
        """
        fig.update_layout(
            title=dict(
                text=title,
                y=0.99,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(
                l=80,   # 왼쪽 여백 (y축 레이블 공간)
                r=20,   # 오른쪽 여백
                t=100,  # 상단 여백 (타이틀 공간)
                b=60    # 하단 여백 (x축 레이블 공간)
            )
        )
    
    def get_qubit_data(self, exp_data: Dict, grid_location: str, 
                      qubit_info: Dict) -> Tuple[Any, str]:
        """특정 큐빗의 데이터 추출
        
        Parameters
        ----------
        exp_data : Dict
            전체 실험 데이터
        grid_location : str
            큐빗의 grid location
        qubit_info : Dict
            큐빗 정보
            
        Returns
        -------
        qubit_data : xr.Dataset
            해당 큐빗의 데이터
        qubit_name : str
            큐빗 이름
        """
        ds_raw = exp_data['ds_raw']
        
        # grid_location에 해당하는 qubit 이름 찾기
        if grid_location not in qubit_info['qubit_mapping']:
            raise KeyError(f"{grid_location} not found in qubit mapping")
        
        qubit_name = qubit_info['qubit_mapping'][grid_location]['qubit_name']
        qubit_dim = qubit_info['dataset_qubit_dim']
        
        # 데이터 접근
        try:
            qubit_dict = {qubit_dim: qubit_name}
            qubit_data = ds_raw.loc[qubit_dict]
        except:
            # 대체 방법 시도
            qubit_data = ds_raw.sel(qubit=qubit_name)
        
        return qubit_data, qubit_name
    
    def calculate_y_range(self, y_data: np.ndarray, 
                         margin_ratio: float = 0.2,
                         min_range: float = 0.0001) -> Tuple[float, float]:
        """Y축 범위 자동 계산
        
        Parameters
        ----------
        y_data : np.ndarray
            Y축 데이터
        margin_ratio : float
            여백 비율 (기본 20%)
        min_range : float
            최소 범위
            
        Returns
        -------
        y_min : float
            Y축 최소값
        y_max : float
            Y축 최대값
        """
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_range = y_max - y_min
        
        # 최소 범위 보장
        if y_range < min_range:
            y_center = (y_max + y_min) / 2
            y_min = y_center - min_range / 2
            y_max = y_center + min_range / 2
        else:
            # 여백 추가
            y_margin = margin_ratio * y_range
            y_min = y_min - y_margin
            y_max = y_max + y_margin
        
        return y_min, y_max
    
    def create_error_trace(self, x_data: Any, y_data: Any, 
                          y_error: Any, name: str, 
                          color: str) -> go.Scatter:
        """에러바가 있는 trace 생성
        
        Parameters
        ----------
        x_data : array-like
            X축 데이터
        y_data : array-like
            Y축 데이터  
        y_error : array-like
            Y축 에러
        name : str
            Trace 이름
        color : str
            색상
            
        Returns
        -------
        go.Scatter
            생성된 trace
        """
        return go.Scatter(
            x=x_data,
            y=y_data,
            error_y=dict(
                type='data',
                array=y_error,
                visible=True,
                color=color,
                thickness=1.5,
                width=3
            ),
            mode='markers+lines',
            name=name,
            line=dict(color=color),
            marker=dict(size=6)
        )
    
    def add_fit_line(self, fig: go.Figure, x_data: Any, y_fit: Any,
                    name: str, color: str, row: int, col: int):
        """Fit 라인 추가
        
        Parameters
        ----------
        fig : go.Figure
            Figure 객체
        x_data : array-like
            X축 데이터
        y_fit : array-like
            Fit된 Y값
        name : str
            라인 이름
        color : str
            색상
        row : int
            서브플롯 행
        col : int
            서브플롯 열
        """
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_fit,
                mode='lines',
                name=name,
                line=dict(
                    color=color,
                    width=2,
                    dash='dash'
                ),
                showlegend=True
            ),
            row=row, 
            col=col
        )
    
    def add_annotation_to_subplot(self, fig: go.Figure, text: str,
                                 x: float, y: float, 
                                 row: int, col: int):
        """서브플롯에 주석 추가
        
        Parameters
        ----------
        fig : go.Figure
            Figure 객체
        text : str
            주석 텍스트
        x : float
            X 위치
        y : float
            Y 위치
        row : int
            서브플롯 행
        col : int
            서브플롯 열
        """
        fig.add_annotation(
            text=text,
            x=x,
            y=y,
            xref=f"x{row*col}",
            yref=f"y{row*col}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            ax=30,
            ay=-30,
            font=dict(size=12)
        )