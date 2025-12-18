import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { fetchOasisSummary, fetchOasisSubjects } from "@/lib/api"

export default async function DatasetPage() {
  const summary = await fetchOasisSummary()
  const subjects = await fetchOasisSubjects(1, 3)

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            OASIS-1 Dataset
          </h2>
          <Badge variant="outline">OASIS-1 only</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Cross-sectional structural MRI dataset with 436 subjects. Each subject
          has T1-weighted MRI and clinical anchors including CDR, MMSE
          (available for a subset), demographics, and derived anatomical
          features.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Subjects</CardTitle>
            <CardDescription>Total OASIS-1 cohort</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">
              {summary.n_subjects}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">CDR labels</CardTitle>
            <CardDescription>Non-missing CDR scores</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">
              {summary.n_cdr_labeled}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              All supervised metrics are computed on labeled subjects only.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">MRI modality</CardTitle>
            <CardDescription>Structural T1-weighted</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              High-resolution T1-weighted images preprocessed and registered to
              a common template for feature extraction.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Example subjects</CardTitle>
            <CardDescription>
              Demographics and clinical anchors (mocked sample)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Subject</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Gender</TableHead>
                  <TableHead>CDR</TableHead>
                  <TableHead>MMSE</TableHead>
                  <TableHead>NWBV</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {subjects.items.map((s) => (
                  <TableRow key={s.subject_id}>
                    <TableCell className="font-mono text-xs">
                      {s.subject_id}
                    </TableCell>
                    <TableCell>{s.age ?? "—"}</TableCell>
                    <TableCell>{s.gender ?? "—"}</TableCell>
                    <TableCell>{s.cdr ?? "—"}</TableCell>
                    <TableCell>{s.mmse ?? "—"}</TableCell>
                    <TableCell>{s.nwbv?.toFixed(2) ?? "—"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Label completeness</CardTitle>
            <CardDescription>Missing-label handling</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • CDR and MMSE scores are not available for all subjects.
              Baseline and prototype models are trained and evaluated on
              subjects with valid labels only.
            </p>
            <p>
              • Unlabeled subjects are still used for describing the dataset but
              are excluded from supervised loss calculations.
            </p>
          </CardContent>
        </Card>
      </section>

      <Alert className="text-xs">
        This portal summarizes the OASIS-1 research dataset. Results are
        provided for transparency and methodological evaluation only and are
        not intended for clinical use.
      </Alert>
    </div>
  )
}


