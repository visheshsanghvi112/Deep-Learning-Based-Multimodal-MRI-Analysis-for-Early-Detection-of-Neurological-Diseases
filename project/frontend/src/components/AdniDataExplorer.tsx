"use client"

import { useState, useEffect, useMemo, memo, ChangeEvent } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Users, Search, Brain, Activity } from "lucide-react"

interface AdniSubject {
    subject_id: string
    diagnosis: string
    gender: string
    age: number
}

interface AdniData {
    dataset: string
    total_subjects: number
    groups: Record<string, number>
    age_range: [number, number]
    subjects: AdniSubject[]
}

const SubjectRow = memo(function SubjectRow({ subject }: { subject: AdniSubject }) {
    const diagnosisColor = {
        CN: "bg-green-500/10 text-green-600 border-green-500/20",
        MCI: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
        AD: "bg-red-500/10 text-red-600 border-red-500/20",
    }[subject.diagnosis] || "bg-gray-500/10 text-gray-600"

    return (
        <TableRow className="hover:bg-muted/50 transition-colors">
            <TableCell className="font-mono text-xs">{subject.subject_id}</TableCell>
            <TableCell>{subject.age}</TableCell>
            <TableCell>{subject.gender}</TableCell>
            <TableCell>
                <Badge className={diagnosisColor}>{subject.diagnosis}</Badge>
            </TableCell>
        </TableRow>
    )
})

export function AdniDataExplorer() {
    const [data, setData] = useState<AdniData | null>(null)
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState("")
    const [diagnosisFilter, setDiagnosisFilter] = useState<string | null>(null)

    useEffect(() => {
        fetch("/adni-data.json")
            .then((res) => res.json())
            .then((json) => {
                setData(json)
                setLoading(false)
            })
            .catch((err) => {
                console.error("Failed to load ADNI data:", err)
                setLoading(false)
            })
    }, [])

    const filteredSubjects = useMemo(() => {
        if (!data) return []
        return data.subjects.filter((s) => {
            const matchesSearch = search === "" ||
                s.subject_id.toLowerCase().includes(search.toLowerCase())
            const matchesDiagnosis = diagnosisFilter === null || s.diagnosis === diagnosisFilter
            return matchesSearch && matchesDiagnosis
        })
    }, [data, search, diagnosisFilter])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-pulse flex items-center gap-2 text-muted-foreground">
                    <Brain className="h-5 w-5" />
                    Loading ADNI data...
                </div>
            </div>
        )
    }

    if (!data) {
        return (
            <div className="text-center text-muted-foreground py-8">
                Failed to load ADNI data
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Stats Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card className="border-l-4 border-l-teal-500">
                    <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                            <Users className="h-6 w-6 text-teal-500" />
                            <div>
                                <div className="text-2xl font-bold">{data.total_subjects}</div>
                                <div className="text-xs text-muted-foreground">Total Subjects</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {Object.entries(data.groups).map(([group, count]) => {
                    const colors = {
                        CN: { border: "border-l-green-500", icon: "text-green-500" },
                        MCI: { border: "border-l-yellow-500", icon: "text-yellow-500" },
                        AD: { border: "border-l-red-500", icon: "text-red-500" },
                    }[group] || { border: "border-l-gray-500", icon: "text-gray-500" }

                    return (
                        <Card key={group} className={`border-l-4 ${colors.border}`}>
                            <CardContent className="pt-4">
                                <div className="flex items-center gap-3">
                                    <Activity className={`h-6 w-6 ${colors.icon}`} />
                                    <div>
                                        <div className="text-2xl font-bold">{count}</div>
                                        <div className="text-xs text-muted-foreground">{group}</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )
                })}
            </div>

            {/* Filters */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                        <Search className="h-5 w-5" />
                        Subject Explorer
                    </CardTitle>
                    <CardDescription>
                        Browse {data.total_subjects} ADNI-1 subjects (showing first 200)
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-3">
                        <div className="flex-1 min-w-[200px]">
                            <input
                                type="text"
                                placeholder="Search by Subject ID..."
                                value={search}
                                onChange={(e: ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
                                className="w-full px-3 py-2 rounded-md border bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                            />
                        </div>
                        <div className="flex gap-2">
                            <Badge
                                variant={diagnosisFilter === null ? "default" : "outline"}
                                className="cursor-pointer"
                                onClick={() => setDiagnosisFilter(null)}
                            >
                                All
                            </Badge>
                            {Object.keys(data.groups).map((group) => (
                                <Badge
                                    key={group}
                                    variant={diagnosisFilter === group ? "default" : "outline"}
                                    className="cursor-pointer"
                                    onClick={() => setDiagnosisFilter(diagnosisFilter === group ? null : group)}
                                >
                                    {group}
                                </Badge>
                            ))}
                        </div>
                    </div>

                    <div className="text-sm text-muted-foreground">
                        Showing {filteredSubjects.length} subjects
                    </div>

                    <div className="max-h-[400px] overflow-auto rounded-lg border">
                        <Table className="min-w-full">
                            <TableHeader className="sticky top-0 bg-card">
                                <TableRow>
                                    <TableHead>Subject ID</TableHead>
                                    <TableHead>Age</TableHead>
                                    <TableHead>Gender</TableHead>
                                    <TableHead>Diagnosis</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {filteredSubjects.slice(0, 50).map((subject) => (
                                    <SubjectRow key={subject.subject_id} subject={subject} />
                                ))}
                            </TableBody>
                        </Table>
                    </div>

                    {filteredSubjects.length > 50 && (
                        <div className="text-center text-sm text-muted-foreground">
                            Showing 50 of {filteredSubjects.length} matching subjects
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
