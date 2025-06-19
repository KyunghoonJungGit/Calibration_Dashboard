"""
File Watcher Module
실험 데이터 폴더를 감시하고 새로운 실험 데이터를 감지하는 모듈
"""
import time
import threading
from pathlib import Path
from typing import Set, Optional
from watchdog.events import FileSystemEventHandler

from .tof_data_loader import ExperimentDataLoader


class ExperimentDataWatcher(FileSystemEventHandler):
    """실험 데이터 폴더를 감시하고 완료된 실험을 감지"""
    
    def __init__(self, dashboard_server, data_loader: Optional[ExperimentDataLoader] = None):
        """
        Parameters
        ----------
        dashboard_server : DashboardServer
            메인 대시보드 서버 인스턴스
        data_loader : ExperimentDataLoader, optional
            데이터 로더 인스턴스. None인 경우 새로 생성
        """
        self.dashboard_server = dashboard_server
        self.data_loader = data_loader or ExperimentDataLoader()
        self.processing: Set[Path] = set()  # 현재 처리 중인 폴더
        self.processed: Set[Path] = set()   # 이미 처리된 폴더
        self.lock = threading.Lock()
        
        # 설정
        self.completion_marker = '.complete'  # 실험 완료 마커 파일
        self.processing_delay = 0.5  # 파일 쓰기 완료 대기 시간 (초)
        self.max_retries = 3  # 최대 재시도 횟수
        self.retry_delay = 1.0  # 재시도 간격 (초)
    
    def on_created(self, event):
        """새 파일/폴더 생성 감지"""
        if event.is_directory:
            return
        
        # 완료 마커 파일 확인
        if event.src_path.endswith(self.completion_marker):
            experiment_dir = Path(event.src_path).parent
            
            with self.lock:
                # 이미 처리 중이거나 처리된 경우 스킵
                if experiment_dir in self.processing or experiment_dir in self.processed:
                    return
                
                self.processing.add(experiment_dir)
            
            # 별도 스레드에서 처리
            thread = threading.Thread(
                target=self._process_experiment_with_retry,
                args=(experiment_dir,),
                daemon=True
            )
            thread.start()
    
    def on_modified(self, event):
        """파일 수정 감지 (일부 시스템에서는 created 대신 modified 이벤트 발생)"""
        if not event.is_directory and event.src_path.endswith(self.completion_marker):
            self.on_created(event)
    
    def _process_experiment_with_retry(self, experiment_dir: Path):
        """재시도 로직을 포함한 실험 폴더 처리"""
        retry_count = 0
        success = False
        
        while retry_count < self.max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"🔄 Retry {retry_count}/{self.max_retries} for {experiment_dir.name}")
                    time.sleep(self.retry_delay)
                
                self._process_experiment_folder(experiment_dir)
                success = True
                
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    print(f"⚠️  Error processing {experiment_dir.name}: {e}")
                    print(f"   Will retry in {self.retry_delay} seconds...")
                else:
                    print(f"❌ Failed to process {experiment_dir.name} after {self.max_retries} attempts")
                    print(f"   Last error: {e}")
                    import traceback
                    traceback.print_exc()
            
        # 처리 완료 후 상태 업데이트
        with self.lock:
            self.processing.discard(experiment_dir)
            if success:
                self.processed.add(experiment_dir)
    
    def _process_experiment_folder(self, experiment_dir: Path):
        """실험 폴더를 처리하여 대시보드에 추가"""
        print(f"\n📁 Processing new experiment: {experiment_dir.name}")
        
        # 파일 쓰기 완료 대기
        time.sleep(self.processing_delay)
        
        # 데이터 로더를 통해 실험 데이터 로드
        experiment_data = self.data_loader.load_experiment(experiment_dir)
        
        if experiment_data:
            # 메타데이터에서 실험 ID 추출
            experiment_id = experiment_data['metadata']['experiment_id']
            
            # 대시보드에 실험 추가
            self.dashboard_server.add_experiment_from_file(
                experiment_data,
                experiment_id
            )
            
            print(f"✅ Successfully added: {experiment_id}")
            self._print_experiment_summary(experiment_data)
        else:
            raise ValueError(f"Failed to load experiment data from {experiment_dir}")
    
    def _print_experiment_summary(self, experiment_data: dict):
        """실험 데이터 요약 출력"""
        metadata = experiment_data['metadata']
        qubit_info = experiment_data['qubit_info']
        
        print(f"   📊 Type: {metadata['experiment_type']}")
        print(f"   🕐 Time: {metadata['timestamp']}")
        print(f"   🔢 Qubits: {len(qubit_info['grid_locations'])}")
        
        # 데이터셋 정보
        dataset_info = metadata.get('dataset_info', {})
        if dataset_info:
            dims = dataset_info.get('dimensions', {})
            if dims:
                dim_str = ", ".join([f"{k}: {v}" for k, v in dims.items()])
                print(f"   📏 Dimensions: {dim_str}")
    
    def get_status(self) -> dict:
        """감시 상태 반환"""
        with self.lock:
            return {
                'processing': len(self.processing),
                'processed': len(self.processed),
                'processing_dirs': list(self.processing),
                'is_idle': len(self.processing) == 0
            }
    
    def reset_processed(self):
        """처리된 폴더 목록 초기화"""
        with self.lock:
            self.processed.clear()
            print("🔄 Reset processed folders list")
    
    def is_experiment_complete(self, experiment_dir: Path) -> bool:
        """실험이 완료되었는지 확인"""
        return (experiment_dir / self.completion_marker).exists()
    
    def mark_experiment_complete(self, experiment_dir: Path):
        """실험을 완료 상태로 표시 (테스트용)"""
        complete_flag = experiment_dir / self.completion_marker
        complete_flag.touch()
        print(f"✓ Marked {experiment_dir.name} as complete")