import { useMemo, useState } from 'react';
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { Download, FileText, Search, Database, Ban } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { formatDate } from '@/lib/utils';
import { TaskResponse, TaskStatus, EntityConfigResponse } from '@/types/data-cleaning';
import { DataTableColumnHeader } from '@/components/shared/tables/DataTableColumnHeader';
import { DataTablePagination } from '@/components/shared/tables/DataTablePagination';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

const statusMap = {
  [TaskStatus.WAITING]: { label: '等待中', variant: 'secondary' as const },
  [TaskStatus.PROCESSING]: { label: '处理中', variant: 'default' as const },
  [TaskStatus.COMPLETED]: { label: '已完成', variant: 'secondary' as const },
  [TaskStatus.FAILED]: { label: '失败', variant: 'destructive' as const },
  [TaskStatus.CANCELLED]: { label: '已取消', variant: 'secondary' as const },
};

interface CleaningTaskListProps {
  tasks: TaskResponse[];
  entityTypes: EntityConfigResponse[];
  onDownload: (taskId: string, fileName: string) => void;
  onCancel?: (taskId: string) => void;
}

export function CleaningTaskList({ tasks, entityTypes, onDownload, onCancel }: CleaningTaskListProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  // 创建实体类型映射
  const entityTypeMap = useMemo(() => {
    return entityTypes.reduce((acc, type) => {
      acc[type.entity_type] = type.display_name;
      return acc;
    }, {} as Record<string, string>);
  }, [entityTypes]);

  const columns = useMemo<ColumnDef<TaskResponse>[]>(() => [
    {
      accessorKey: 'created_at',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title="创建时间" />
      ),
      cell: ({ row }) => (
        <div>{formatDate(row.original.created_at)}</div>
      ),
    },
    {
      accessorKey: 'entity_type',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title="实体类型" />
      ),
      cell: ({ row }) => (
        <div className="flex items-center gap-2">
          <span className="font-medium">
            {entityTypeMap[row.original.entity_type] || row.original.entity_type}
          </span>
        </div>
      ),
    },
    {
      id: 'features',
      header: "功能特性",
      cell: ({ row }) => {
        const task = row.original;
        return (
          <div className="flex items-center gap-2">
            <Badge variant={task.search_enabled === 'enabled' ? 'default' : 'secondary'}>
              <Search className="mr-1 h-3 w-3" />
              搜索{task.search_enabled === 'enabled' ? '已启用' : '已禁用'}
            </Badge>
            <Badge variant={task.retrieval_enabled === 'enabled' ? 'default' : 'secondary'}>
              <Database className="mr-1 h-3 w-3" />
              检索{task.retrieval_enabled === 'enabled' ? '已启用' : '已禁用'}
            </Badge>
          </div>
        );
      },
    },
    {
      accessorKey: 'status',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title="状态" />
      ),
      cell: ({ row }) => {
        const status = statusMap[row.original.status];
        return (
          <div className="flex items-center gap-2">
            <Badge variant={status.variant}>{status.label}</Badge>
            {row.original.error_message && (
              <span className="text-sm text-muted-foreground truncate max-w-[200px]">
                {row.original.error_message}
              </span>
            )}
          </div>
        );
      },
    },
    {
      accessorKey: 'total_records',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title="总行数" />
      ),
      cell: ({ row }) => {
        const task = row.original;
        return (
          <div className="w-[80px] text-right">
            {task.total_records?.toLocaleString() || '-'}
          </div>
        );
      },
    },
    {
      accessorKey: 'progress',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title="进度" />
      ),
      cell: ({ row }) => {
        const task = row.original;
        const processed = task.processed_records || 0;
        const total = task.total_records || 0;
        const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

        return (
          <div className="w-[120px]">
            <Progress value={percentage} />
          </div>
        );
      },
    },
    {
      id: 'actions',
      cell: ({ row }) => {
        const task = row.original;
        const canDownload = task.status === TaskStatus.COMPLETED;
        const canCancel = onCancel && (task.status === TaskStatus.WAITING || task.status === TaskStatus.PROCESSING);

        return (
          <div className="flex items-center gap-2">
            {canDownload && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDownload(
                  task.task_id,
                  `清洗结果_${formatDate(task.created_at)}`
                )}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                <span>下载</span>
              </Button>
            )}
            {canCancel && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onCancel(task.task_id)}
                className="flex items-center gap-2 text-destructive hover:text-destructive"
              >
                <Ban className="h-4 w-4" />
                <span>取消</span>
              </Button>
            )}
          </div>
        );
      },
    },
  ], [onDownload, onCancel, entityTypeMap]);

  const table = useReactTable({
    data: tasks,
    columns,
    state: {
      sorting,
      columnFilters,
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  暂无任务数据
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <DataTablePagination table={table} />
    </div>
  );
}