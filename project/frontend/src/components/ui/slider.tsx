"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

const Slider = React.forwardRef<
    HTMLInputElement,
    Omit<React.InputHTMLAttributes<HTMLInputElement>, "value"> & {
        value: number[]
        onValueChange?: (value: number[]) => void
    }
>(({ className, value, min, max, step, onValueChange, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = parseFloat(e.target.value)
        onValueChange?.([val])
    }

    return (
        <div className="relative flex w-full touch-none select-none items-center">
            <input
                type="range"
                className={cn(
                    "h-2 w-full cursor-pointer appearance-none rounded-full bg-secondary disabled:cursor-not-allowed disabled:opacity-50",
                    "accent-primary",
                    className
                )}
                ref={ref}
                value={Array.isArray(value) ? value[0] : value}
                min={min}
                max={max}
                step={step}
                onChange={handleChange}
                {...props}
            />
        </div>
    )
})
Slider.displayName = "Slider"

export { Slider }
