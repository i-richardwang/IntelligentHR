import { useState, useCallback } from 'react';
import { ClassificationTaskResponse } from '@/types/api';
import { textClassificationApi } from '@/services';
import { useToast } from '@/hooks/use-toast';

export function useClassificationTaskList() {
  const [tasks, setTasks] = useState<ClassificationTaskResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { toast } = useToast();

  const fetchTasks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await textClassificationApi.getTasks();

      if (response.data) {
        setTasks(response.data);
      }
    } catch (error) {
      console.error('Classification task list error:', error);
      const errorMessage = error instanceof Error ? error.message : '请稍后重试';
      setError(new Error(errorMessage));
      toast({
        variant: 'destructive',
        title: '获取任务列表失败',
        description: errorMessage,
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const createTask = useCallback(async (
    file: File,
    context: string,
    categories: Record<string, string>,
    is_multi_label: boolean
  ) => {
    try {
      setLoading(true);
      const response = await textClassificationApi.createTask({
        file,
        context,
        categories,
        is_multi_label
      });

      if (response.data) {
        toast({
          title: '创建成功',
          description: '分类任务已开始处理',
        });
        await fetchTasks(); // 刷新任务列表
        return response.data;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '创建任务失败';
      toast({
        variant: 'destructive',
        title: '创建失败',
        description: errorMessage,
      });
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchTasks, toast]);

  const downloadResult = useCallback(async (taskId: string, fileName: string) => {
    try {
      await textClassificationApi.downloadResult(taskId, fileName);
    } catch (error) {
      console.error('Download failed:', error);
      toast({
        variant: 'destructive',
        title: '下载失败',
        description: error instanceof Error ? error.message : '请稍后重试',
      });
    }
  }, [toast]);

  const cancelTask = useCallback(async (taskId: string) => {
    try {
      await textClassificationApi.cancelTask(taskId);
      toast({
        title: '已发送取消请求',
        description: '任务将在当前处理完成后停止',
      });
      // 刷新任务列表
      await fetchTasks();
    } catch (error) {
      toast({
        variant: 'destructive',
        title: '取消失败',
        description: error instanceof Error ? error.message : '请稍后重试',
      });
    }
  }, [fetchTasks, toast]);

  return {
    tasks,
    loading,
    error,
    fetchTasks,
    createTask,
    downloadResult,
    cancelTask,
  };
}