"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { ArrowRightLeft } from "lucide-react"

export function RobustnessBanner() {
    const [count1, setCount1] = useState(0)
    const [count2, setCount2] = useState(0)

    useEffect(() => {
        // Simple counter animation
        const duration = 1500
        const start = performance.now()

        const animate = (time: number) => {
            const progress = Math.min((time - start) / duration, 1)
            const ease = 1 - Math.pow(1 - progress, 3) // Cubic ease out

            setCount1(0.607 * ease)
            setCount2(0.624 * ease)

            if (progress < 1) {
                requestAnimationFrame(animate)
            }
        }

        requestAnimationFrame(animate)
    }, [])

    return (
        <motion.section
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="relative overflow-hidden rounded-xl border bg-gradient-to-br from-blue-500/10 via-purple-500/5 to-teal-500/10 p-1"
        >
            <div className="rounded-lg bg-background/40 backdrop-blur-sm p-6 md:p-8">
                <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-purple-600">
                            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-500/10 ring-1 ring-purple-500/20">
                                <ArrowRightLeft className="h-4 w-4" />
                            </span>
                            <span className="font-semibold tracking-tight">Key Finding: Cross-Dataset Robustness</span>
                        </div>
                        <p className="max-w-xl text-sm text-muted-foreground leading-relaxed">
                            We observed a distinct <span className="font-medium text-foreground">direction-dependent robustness</span>.
                            MRI-Only models generalize better when transferring from OASIS to ADNI, while Late Fusion is superior
                            in the reverse direction.
                        </p>
                    </div>

                    <div className="flex items-center gap-8 md:gap-12">
                        <div className="text-center">
                            <div className="text-3xl font-bold tracking-tighter text-foreground tabular-nums">
                                {count1.toFixed(3)}
                            </div>
                            <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mt-1">
                                OASIS → ADNI
                            </div>
                        </div>

                        <div className="h-12 w-px bg-border/60" />

                        <div className="text-center">
                            <div className="text-3xl font-bold tracking-tighter text-foreground tabular-nums">
                                {count2.toFixed(3)}
                            </div>
                            <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mt-1">
                                ADNI → OASIS
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </motion.section>
    )
}
