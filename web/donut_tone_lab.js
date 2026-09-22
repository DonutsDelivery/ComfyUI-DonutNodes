import { app } from '../../scripts/app.js';
import { addToneLabToV5 } from './donut_tone_lab_model.js';

app.registerExtension({
    name:'Donut.ToneLabV5',
    beforeConfigureGraph(data) {
        const report = addToneLabToV5(data);
        if (report.changed) console.info('[Donut Tone Lab] Added disabled final-tone stage. Select a v4 model in Save images.');
        else if (data?.extra?.donut_workflow?.release === 'V5' && !['already_migrated','already_present'].includes(report.reason)) {
            console.info(`[Donut Tone Lab] Automatic wiring skipped (${report.reason}); custom graphs can use Donut Tone Lab manually.`);
        }
    },
});
