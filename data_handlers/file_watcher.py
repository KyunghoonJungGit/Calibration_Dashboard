"""
File Watcher Module
실험 데이터 폴더를 감시하고 새로운 실험 데이터를 감지하는 모듈
Modified to work without separate data loaders
"""

import time
import threading
from pathlib import Path
from typing import Set
from watchdog.events import FileSystemEventHandler


class ExperimentDataWatcher(FileSystemEventHandler):
    """실험 데이터 폴더를 감시하고 완료된 실험을 감지"""
    
    def __init__(self, dashboard_server):
        """
        Parameters
        ----------
        dashboard_server : DashboardServer
            메인 대시보드 서버 인스턴스
        """
        self.dashboard_server = dashboard_server
        self.processing: Set[Path] = set()  # 현재 처리 중인 폴더
        self.processed: Set[Path] = set()   # 이미 처리된 폴더
        self.lock = threading.Lock()
        
        # 설정
        self.completion_marker = '.complete'  # 실험 완료 마커 파일
        self.processing_delay = 0.5  # 파일 쓰기 완료 대기 시간 (초)
    
    def on_created(self, event):
        """새 파일/폴더 생성 감지"""
        if event.is_directory:
            return
        
        # 완료 마커 파일 확인
        if event.src_path.endswith(self.completion_marker):
            experiment_dir = Path(event.src_path).parent
            
            # 날짜 폴더 내부에 있는지 확인
            # 구조: dashboard_data/2025-06-17/#9_02a_resonator_spectroscopy_060916/.complete
            if experiment_dir.parent.name.startswith('20'):  # 날짜 폴더인지 확인
                with self.lock:
                    # 이미 처리 중이거나 처리된 경우 스킵
                    if experiment_dir in self.processing or experiment_dir in self.processed:
                        return
                    
                    self.processing.add(experiment_dir)
                
                # 별도 스레드에서 처리
                thread = threading.Thread(
                    target=self._process_experiment_folder,
                    args=(experiment_dir,),
                    daemon=True
                )
                thread.start()
    
    def on_modified(self, event):
        """파일 수정 감지 (일부 시스템에서는 created 대신 modified 이벤트 발생)"""
        if not event.is_directory and event.src_path.endswith(self.completion_marker):
            self.on_created(event)
    
    def _process_experiment_folder(self, experiment_dir: Path):
        """실험 폴더를 처리하여 대시보드에 추가"""
        print(f"\n📁 Processing new experiment: {experiment_dir.name}")
        
        # 파일 쓰기 완료 대기
        time.sleep(self.processing_delay)
        
        try:
            # 대시보드에 실험 디렉토리 추가
            # 실제 데이터 로딩은 각 plotter에서 수행
            self.dashboard_server.add_experiment_from_directory(experiment_dir)
            
            print(f"✅ Successfully added: {experiment_dir.name}")
            
        except Exception as e:
            print(f"❌ Failed to add {experiment_dir.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 처리 완료 후 상태 업데이트
            with self.lock:
                self.processing.discard(experiment_dir)
                self.processed.add(experiment_dir)
    
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