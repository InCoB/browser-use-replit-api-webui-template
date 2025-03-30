import * as React from "react";
import { cn } from "@/lib/utils";

interface BarProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  label: string;
  color?: string;
}

export function BarChart({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("space-y-8", className)} {...props}>
      {children}
    </div>
  );
}

export function Bar({
  className,
  value,
  max = 100,
  label,
  color = "bg-primary",
  ...props
}: BarProps) {
  const percentage = (value / max) * 100;
  
  return (
    <div className="flex items-center gap-4" {...props}>
      <div className="w-24 text-right text-sm font-medium">{label}</div>
      <div className="flex-1">
        <div className="h-8 w-full overflow-hidden rounded bg-gray-200">
          <div
            className={cn(
              "flex h-full items-center justify-end px-2 text-white",
              color
            )}
            style={{ width: `${percentage}%` }}
          >
            <span className="text-sm font-semibold">{value}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
